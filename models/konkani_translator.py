"""
Custom Translation Model for Konkani → English
Seq2Seq with Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    """BiLSTM Encoder"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_lengths):
        """
        Args:
            src: (batch, src_len)
            src_lengths: (batch,)
        Returns:
            outputs: (batch, src_len, hidden_dim*2)
            hidden: tuple of (num_layers*2, batch, hidden_dim)
        """
        embedded = self.dropout(self.embedding(src))
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        outputs, hidden = self.lstm(packed)
        
        # Unpack
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        return outputs, hidden


class Attention(nn.Module):
    """Bahdanau Attention"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs, mask=None):
        """
        Args:
            hidden: (batch, hidden_dim)
            encoder_outputs: (batch, src_len, hidden_dim*2)
            mask: (batch, src_len)
        Returns:
            attention_weights: (batch, src_len)
        """
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        
        # Repeat hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Calculate attention scores
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        # Apply mask
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    """LSTM Decoder with Attention"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.attention = Attention(hidden_dim)
        
        self.lstm = nn.LSTM(
            embedding_dim + hidden_dim * 2,  # embedding + context
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim * 3 + embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, encoder_outputs, mask=None):
        """
        Args:
            input: (batch,) - current token
            hidden: tuple of (num_layers, batch, hidden_dim)
            encoder_outputs: (batch, src_len, hidden_dim*2)
            mask: (batch, src_len)
        Returns:
            output: (batch, vocab_size)
            hidden: updated hidden state
        """
        input = input.unsqueeze(1)  # (batch, 1)
        embedded = self.dropout(self.embedding(input))  # (batch, 1, embedding_dim)
        
        # Calculate attention
        attn_weights = self.attention(hidden[0][-1], encoder_outputs, mask)
        attn_weights = attn_weights.unsqueeze(1)  # (batch, 1, src_len)
        
        # Apply attention to encoder outputs
        context = torch.bmm(attn_weights, encoder_outputs)  # (batch, 1, hidden_dim*2)
        
        # Concatenate embedding and context
        lstm_input = torch.cat((embedded, context), dim=2)
        
        # LSTM step
        output, hidden = self.lstm(lstm_input, hidden)
        
        # Prediction
        output = output.squeeze(1)  # (batch, hidden_dim)
        context = context.squeeze(1)  # (batch, hidden_dim*2)
        embedded = embedded.squeeze(1)  # (batch, embedding_dim)
        
        prediction = self.fc(torch.cat((output, context, embedded), dim=1))
        
        return prediction, hidden


class Seq2SeqTranslator(nn.Module):
    """Complete Seq2Seq Translation Model"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim=256, 
                 hidden_dim=512, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.encoder = Encoder(src_vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        
        # Bridge to convert bidirectional encoder hidden to decoder hidden
        self.bridge = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=0.5):
        """
        Args:
            src: (batch, src_len)
            src_lengths: (batch,)
            tgt: (batch, tgt_len)
            teacher_forcing_ratio: probability of using teacher forcing
        Returns:
            outputs: (batch, tgt_len, vocab_size)
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.vocab_size
        
        # Encode
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # Convert encoder hidden to decoder hidden
        # hidden[0] shape: (num_layers*2, batch, hidden_dim)
        # We need: (num_layers, batch, hidden_dim)
        h = hidden[0].view(self.encoder.lstm.num_layers, 2, batch_size, -1)
        h = torch.cat([h[:, 0, :, :], h[:, 1, :, :]], dim=2)  # Concatenate forward and backward
        h = self.bridge(h)
        
        c = hidden[1].view(self.encoder.lstm.num_layers, 2, batch_size, -1)
        c = torch.cat([c[:, 0, :, :], c[:, 1, :, :]], dim=2)
        c = self.bridge(c)
        
        hidden = (h, c)
        
        # Create mask for attention
        mask = (src != 0).long()
        
        # Decode
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(src.device)
        
        # First input is <SOS> token
        input = tgt[:, 0]
        
        for t in range(1, tgt_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs, mask)
            outputs[:, t, :] = output
            
            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = tgt[:, t] if teacher_force else top1
        
        return outputs
    
    def translate(self, src, src_lengths, max_len=100, sos_token=2, eos_token=3):
        """
        Translate without teacher forcing (inference)
        
        Args:
            src: (batch, src_len)
            src_lengths: (batch,)
            max_len: maximum translation length
            sos_token: start of sequence token ID
            eos_token: end of sequence token ID
        Returns:
            translations: (batch, max_len)
        """
        self.eval()
        batch_size = src.size(0)
        
        with torch.no_grad():
            # Encode
            encoder_outputs, hidden = self.encoder(src, src_lengths)
            
            # Convert encoder hidden to decoder hidden
            h = hidden[0].view(self.encoder.lstm.num_layers, 2, batch_size, -1)
            h = torch.cat([h[:, 0, :, :], h[:, 1, :, :]], dim=2)
            h = self.bridge(h)
            
            c = hidden[1].view(self.encoder.lstm.num_layers, 2, batch_size, -1)
            c = torch.cat([c[:, 0, :, :], c[:, 1, :, :]], dim=2)
            c = self.bridge(c)
            
            hidden = (h, c)
            
            # Create mask
            mask = (src != 0).long()
            
            # Start with <SOS>
            input = torch.full((batch_size,), sos_token, dtype=torch.long).to(src.device)
            
            translations = []
            
            for _ in range(max_len):
                output, hidden = self.decoder(input, hidden, encoder_outputs, mask)
                top1 = output.argmax(1)
                translations.append(top1.unsqueeze(1))
                input = top1
                
                # Stop if all sequences have generated <EOS>
                if (top1 == eos_token).all():
                    break
            
            translations = torch.cat(translations, dim=1)
        
        return translations


def create_translator(src_vocab_size, tgt_vocab_size, **kwargs):
    """Factory function to create translator"""
    model = Seq2SeqTranslator(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        **kwargs
    )
    return model

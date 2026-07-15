import pandas as pd
import editdistance

df = pd.read_excel('outputs/predictions_analysis_val.xlsx', sheet_name='Detailed Predictions')

total_ref_words = 0
total_ref_chars = 0

mamba_sub = df['SM'].sum()
mamba_del = df['DM'].sum()
mamba_ins = df['IM'].sum()
mamba_word_errors = mamba_sub + mamba_del + mamba_ins
mamba_char_errors = 0

gram3_sub = df['S3'].sum()
gram3_del = df['D3'].sum()
gram3_ins = df['I3'].sum()
gram3_word_errors = gram3_sub + gram3_del + gram3_ins
gram3_char_errors = 0

gram4_sub = df['S4'].sum()
gram4_del = df['D4'].sum()
gram4_ins = df['I4'].sum()
gram4_word_errors = gram4_sub + gram4_del + gram4_ins
gram4_char_errors = 0

for idx, row in df.iterrows():
    ref = str(row['Reference'])
    if ref == 'nan': ref = ''
    
    hyp_m = str(row['Mamba Hyp']) if str(row['Mamba Hyp']) != 'nan' else ''
    hyp_3 = str(row['3-gram Hyp']) if str(row['3-gram Hyp']) != 'nan' else ''
    hyp_4 = str(row['4-gram Hyp']) if str(row['4-gram Hyp']) != 'nan' else ''
    
    ref_words = ref.split()
    total_ref_words += len(ref_words)
    total_ref_chars += len(ref)
    
    mamba_char_errors += editdistance.eval(ref, hyp_m)
    gram3_char_errors += editdistance.eval(ref, hyp_3)
    gram4_char_errors += editdistance.eval(ref, hyp_4)

wer_m = (mamba_word_errors / total_ref_words) * 100 if total_ref_words > 0 else 0
cer_m = (mamba_char_errors / total_ref_chars) * 100 if total_ref_chars > 0 else 0

wer_3 = (gram3_word_errors / total_ref_words) * 100 if total_ref_words > 0 else 0
cer_3 = (gram3_char_errors / total_ref_chars) * 100 if total_ref_chars > 0 else 0

wer_4 = (gram4_word_errors / total_ref_words) * 100 if total_ref_words > 0 else 0
cer_4 = (gram4_char_errors / total_ref_chars) * 100 if total_ref_chars > 0 else 0

print("==== 3-GRAM ====")
print(f"WER: {wer_3:.2f}% | CER: {cer_3:.2f}%")
print("==== 4-GRAM ====")
print(f"WER: {wer_4:.2f}% | CER: {cer_4:.2f}%")
print("==== MAMBA ====")
print(f"WER: {wer_m:.2f}% | CER: {cer_m:.2f}%")

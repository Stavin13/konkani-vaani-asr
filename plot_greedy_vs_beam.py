#!/usr/bin/env python3
"""
Plot greedy vs beam search comparison from best_conformer_ctc_greedy_vs_beam.json
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

with open("best_conformer_ctc_greedy_vs_beam.json") as f:
    data = json.load(f)

g = data["greedy"]
b = data["beam"]
beam_width = data["beam_width"]
samples = data["samples"]
avg_greedy_ms = data["avg_greedy_ms"]
avg_beam_ms = data["avg_beam_ms"]

# ── colour palette ────────────────────────────────────────────────────────────
C_GREEDY = "#4C9BE8"   # blue
C_BEAM   = "#F28B30"   # orange
C_BG     = "#0F1117"
C_PANEL  = "#1A1D27"
C_TEXT   = "#E8EAF0"
C_GRID   = "#2A2D3A"

fig = plt.figure(figsize=(13, 8), facecolor=C_BG)
fig.suptitle(
    f"Greedy  vs  Beam Search (w={beam_width})  ·  best_conformer_ctc.pt  ·  {samples} test samples",
    color=C_TEXT, fontsize=13, fontweight="bold", y=0.97,
)

gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
                      left=0.08, right=0.96, top=0.91, bottom=0.09)

# ── helper ────────────────────────────────────────────────────────────────────
def styled_ax(ax, title):
    ax.set_facecolor(C_PANEL)
    ax.spines[:].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT, labelsize=9)
    ax.set_title(title, color=C_TEXT, fontsize=10, fontweight="bold", pad=8)
    ax.yaxis.label.set_color(C_TEXT)
    ax.grid(axis="y", color=C_GRID, linewidth=0.8, linestyle="--")
    return ax


def bar_pair(ax, greedy_val, beam_val, ylabel, lower_better=True):
    styled_ax(ax, ylabel)
    x = np.array([0, 1])
    bars = ax.bar(x, [greedy_val, beam_val],
                  color=[C_GREEDY, C_BEAM],
                  width=0.45, edgecolor="none", zorder=3)
    # value labels on top of bars
    for bar, val in zip(bars, [greedy_val, beam_val]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(greedy_val, beam_val) * 0.015),
                f"{val:.2f}%", ha="center", va="bottom",
                color=C_TEXT, fontsize=10, fontweight="bold")

    # delta arrow annotation
    delta = beam_val - greedy_val
    better = (delta < 0) if lower_better else (delta > 0)
    sign = "−" if delta < 0 else "+"
    color = "#5EE87B" if better else "#E85E5E"
    ax.annotate(
        f"Δ {sign}{abs(delta):.2f}%", xy=(1, max(greedy_val, beam_val) * 0.5),
        ha="center", va="center", color=color, fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc=C_BG, ec=color, lw=1.2),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["Greedy", f"Beam (w={beam_width})"], color=C_TEXT, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    top = max(greedy_val, beam_val)
    ax.set_ylim(0, top * 1.25)


# ── Subplot 1: WER ────────────────────────────────────────────────────────────
bar_pair(fig.add_subplot(gs[0, 0]), g["wer"], b["wer"],
         "Word Error Rate (WER %)", lower_better=True)

# ── Subplot 2: CER ────────────────────────────────────────────────────────────
bar_pair(fig.add_subplot(gs[0, 1]), g["cer"], b["cer"],
         "Char Error Rate (CER %)", lower_better=True)

# ── Subplot 3: Word Accuracy ──────────────────────────────────────────────────
bar_pair(fig.add_subplot(gs[1, 0]), g["word_acc"], b["word_acc"],
         "Word Accuracy (%)", lower_better=False)

# ── Subplot 4: Speed comparison (ms, log scale) ───────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
styled_ax(ax4, "Avg Decode Time per Utterance")
bars = ax4.bar([0, 1], [avg_greedy_ms, avg_beam_ms],
               color=[C_GREEDY, C_BEAM],
               width=0.45, edgecolor="none", zorder=3)
ax4.set_yscale("log")
ax4.set_ylabel("ms  (log scale)", fontsize=9)
ax4.set_xticks([0, 1])
ax4.set_xticklabels(["Greedy", f"Beam (w={beam_width})"], color=C_TEXT, fontsize=10)
for bar, val in zip(bars, [avg_greedy_ms, avg_beam_ms]):
    ax4.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() * 1.5,
             f"{val:.1f} ms", ha="center", va="bottom",
             color=C_TEXT, fontsize=10, fontweight="bold")
speedup_label = f"{avg_beam_ms / avg_greedy_ms:.0f}× slower"
ax4.text(0.5, 0.85, speedup_label, transform=ax4.transAxes,
         ha="center", color="#E85E5E", fontsize=10, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", fc=C_BG, ec="#E85E5E", lw=1.2))

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=C_GREEDY, label="Greedy decode"),
    mpatches.Patch(color=C_BEAM,   label=f"Beam search  (width={beam_width})"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2,
           frameon=True, facecolor=C_BG, edgecolor=C_GRID,
           labelcolor=C_TEXT, fontsize=10, bbox_to_anchor=(0.5, 0.01))

out = "greedy_vs_beam_comparison.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=C_BG)
print(f"✅  Saved → {out}")

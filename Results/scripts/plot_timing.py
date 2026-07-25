"""
plot_timing.py
Generates a professional line plot from optimized_c_timing_results_*.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Data─────────
if len(sys.argv) < 2:
    print("Usage: python plot_timing.py <csv_file>")
    sys.exit(1)

CSV_FILE = sys.argv[1]
df = pd.read_csv(CSV_FILE)


# Style────────
BACKGROUND   = "#0f1117"
GRID_COLOR   = "#1e2130"
ACCENT       = "#4fc3f7"
FILL_COLOR   = "#4fc3f7"
TEXT_COLOR   = "#e0e6f0"
MUTED        = "#7a8ba0"
GRID_COLOR   = "#1e2130"

plt.rcParams.update({
    "font.family":       "monospace",
    "axes.facecolor":    BACKGROUND,
    "figure.facecolor":  BACKGROUND,
    "text.color":        TEXT_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "axes.edgecolor":    GRID_COLOR,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    0.8,
    "grid.alpha":        1.0,
})

# Figure───────
fig, ax = plt.subplots(figsize=(10, 6))

N = df["channels"]   # channels = BASE_C * N = 1*N, so channels == N
x = N
print(df)
# Shaded min/max band
ax.fill_between(
    x, df["min_ms"], df["max_ms"],
    color=FILL_COLOR, alpha=0.12, label="Min / Max range",
)

# Average line
ax.plot(
    x, df["avg_ms"],
    color=ACCENT, linewidth=2.5, marker="o", markersize=7,
    markerfacecolor=BACKGROUND, markeredgecolor=ACCENT, markeredgewidth=2,
    label="Average",
    zorder=3,
)

# Annotate each point with the avg value
for _, row in df.iterrows():
    ax.annotate(
        f"{row['avg_ms']:.2f} ms",
        xy=(row["tokens"], row["avg_ms"]),
        xytext=(8, 6), textcoords="offset points",
        fontsize=8.5, color=ACCENT, alpha=0.85,
    )

# Axes labels & title
impl  = df["implementation"].iloc[0]
if impl == "C++":
    impl = "Baseline C++"
batch = int(df["batch"].iloc[0])

ax.set_title(
    f"{impl} — Timing vs. Size N  (batch size {batch})",
    fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=16,
)
ax.set_xlabel("N", fontsize=11, labelpad=10)
ax.set_ylabel("Time (ms)", fontsize=11, labelpad=10)

# X-axis: show every actual token value
ax.set_xticks(x)
ax.set_xticklabels([str(v) for v in x], fontsize=9)

# Y-axis: clean tick formatting
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f ms"))
ax.tick_params(axis="both", length=4)



# Grid & spines
ax.grid(axis="y", linestyle="--")
ax.grid(axis="x", linestyle=":", alpha=0.4)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID_COLOR)

# Legend───────
ax.legend(
    frameon=True, facecolor="#1a1f2e", edgecolor=GRID_COLOR,
    labelcolor=TEXT_COLOR, fontsize=9, loc="upper left",
)

ax.text(
    0.3, 0.97,
    "N: batch=8,  tokens=2N,  channels=N,  layers=N",
    transform=ax.transAxes, fontsize=8, color=MUTED,
    va="top", ha="left",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1f2e", edgecolor=GRID_COLOR, alpha=0.8),
)

# Save─────────
OUT = "timing_plot.png"
plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Saved → {OUT}")
plt.show()

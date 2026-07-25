"""
plot_timing_comparison.py
Generates a professional line plot comparing three CSV runs
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Data
CSV_FILES = {
    "baseline c++": "c_timing_results_2026-05-25_22-17-15.csv",
    "optimized c++": "optimized_c_timing_results_2026-05-30_20-20-13.csv",
    #"python code": "python_timing_results_2026-05-30_20-11-29.csv",
}

dfs = {}
for run_name, csv_file in CSV_FILES.items():
    df = pd.read_csv(csv_file)

    # Ensure we have millisecond columns. Fall back from seconds if needed
    if "avg_ms" not in df.columns and "avg_s" in df.columns:
        df["avg_ms"] = df["avg_s"] * 1000.0
        df["min_ms"] = df["min_s"] * 1000.0
        df["max_ms"] = df["max_s"] * 1000.0
    # If CSV uses avg_ms already, keep as-is. Ensure numeric dtype
    for col in ("avg_ms", "min_ms", "max_ms"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    dfs[run_name] = df
print("Loaded CSV files:")
for run_name, df in dfs.items():
    print(f"  {run_name}: {len(df)} points")
# print(dfs)

# Style
BACKGROUND   = "#0f1117"
GRID_COLOR   = "#1e2130"
TEXT_COLOR   = "#e0e6f0"
MUTED        = "#7a8ba0"

# Color palette for three runs
COLORS = {
    "baseline c++": "#4fc3f7",  # blaue
    "optimized c++": "#22B911",  # green
    #"python code": "#e36a23",  # orange
}

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

# Figure
fig, ax = plt.subplots(figsize=(12, 7))

impl  = dfs["baseline c++"]["implementation"].iloc[0]
batch = int(dfs["baseline c++"]["batch"].iloc[0])

N_values = dfs["optimized c++"]["channels"].values  # channels = BASE_C*N = N

# Plot each run
for run_name, df in dfs.items():
    print(f"Plotting {run_name} with {len(df)} points...")
    x = df["channels"]   # == N
    color = COLORS.get(run_name, "#cccccc")
    
    # Average line (milliseconds)
    ax.plot(
        x, df["avg_ms"],
        color=color, linewidth=2.5, marker="o", markersize=7,
        markerfacecolor=BACKGROUND, markeredgecolor=color, markeredgewidth=2,
        label=run_name.upper(),
        zorder=3,
    )
    
    # Shaded min/max band (lighter)
    if "min_ms" in df.columns and "max_ms" in df.columns:
        ax.fill_between(
            x, df["min_ms"], df["max_ms"],
            color=color, alpha=0.08,
        )
    
    # Annotate each point with the avg value (ms)
    

# Axes labels & title
ax.set_title(
    f"Baseline c++ — and optimized c++ timing Comparison (batch size {batch})",
    fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=16,
)
ax.set_xlabel("N", fontsize=11, labelpad=10)
ax.set_ylabel("Time (ms)", fontsize=11, labelpad=10)

# X-axis: show N values
ax.set_xticks(N_values)
ax.set_xticklabels([str(v) for v in N_values], fontsize=9)

# Y-axis: show milliseconds and finer ticks
ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8, prune='both'))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, pos: f"{int(v)} ms"))
ax.tick_params(axis="both", length=4)

# Grid & spines
ax.grid(axis="y", linestyle="--")
ax.grid(axis="x", linestyle=":", alpha=0.4)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID_COLOR)

# Legend
ax.legend(
    frameon=True, facecolor="#1a1f2e", edgecolor=GRID_COLOR,
    labelcolor=TEXT_COLOR, fontsize=10, loc="upper left",
    ncol=3, columnspacing=1.0,
)

ax.text(
    0.68, 0.97,
    "N: batch=8,  tokens=2N,  channels=N,  layers=N",
    transform=ax.transAxes, fontsize=8, color=MUTED,
    va="top", ha="left",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1f2e", edgecolor=GRID_COLOR, alpha=0.8),
)

# Save
OUT = "timing_comparison_opt_bas.png"
plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Saved → {OUT}")
plt.show()
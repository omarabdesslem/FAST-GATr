
from __future__ import annotations
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM80 = os.path.abspath(os.path.join(HERE, "..", ".."))
CODE_PYTHON = os.path.join(TEAM80, "Code", "python")
sys.path.insert(0, CODE_PYTHON)
import tools.cpp_flop_counter as fc

E2E = os.path.join(TEAM80, "Results", "timing_results", "CSVs", "M1", "end_to_end")
PK = os.path.join(TEAM80, "Results", "timing_results", "CSVs", "M1", "per_kernel")
OUT = os.path.join(TEAM80, "Report", "Images")
os.makedirs(OUT, exist_ok=True)

E2E_FILES = {
    "Baseline C++": "c_timing_results_2026-05-29_12-17-11.csv",
    "Optimized C++ (scalar)": "optimized_for_cache_c_timing_results_2026-05-30_15-06-18.csv",
    "SIMD C++": "simd_c_timing_results_2026-05-31_17-05-27.csv",
    "Python (EzGATr)": "python_timing_results_2026-05-30_13-27-38.csv",
}
PK_FILES = {
    "baseline": "baseline_c_per_kernel_seconds_results_2026-06-01_08-01-50.csv",
    "scalar":   "optimized_for_cache_c_per_kernel_seconds_results_2026-06-01_08-48-04.csv",
    "simd":     "simd_c_per_kernel_seconds_results_2026-06-01_08-35-31.csv",
}
COLORS = {"Baseline C++": "tab:red", "Optimized C++ (scalar)": "tab:orange",
          "SIMD C++": "tab:green", "Python (EzGATr)": "tab:blue"}


def load_e2e():
    d = {}
    for label, fn in E2E_FILES.items():
        df = pd.read_csv(os.path.join(E2E, fn)).sort_values("channels")
        d[label] = df
    return d


def e2e_flops(b, t, c, l):
    """Common sparse flop model: layers x (one of each op), unweighted op count."""
    B = fc.Build.OPTIMIZED
    per_block = (
        fc.equi_linear_count(fc.EquiLinearCfg(b, t, c, c, True, B)).flops_total()
        + fc.equi_rms_norm_count(fc.EquiRMSNormCfg(n=b * t, channels=c, build=B)).flops_total()
        + fc.geometric_product_count(fc.GeometricProductCfg(b * t * c, B)).flops_total()
        + fc.equi_join_count(fc.EquiJoinCfg(m=b * t * c, build=B)).flops_total()
        + fc.scaler_gated_gelu_count(fc.ScalerGatedGeluCfg(b * t * c, B)).flops_total()
        + fc.equi_geometric_attention_count(
            fc.EquiGeometricAttentionCfg(batch=b, heads=1, t_query=t, t_key=t,
                                         channels_qk=c, channels_v=c,
                                         kinds=["ipa", "daa"], build=B)).flops_total()
    )
    return l * per_block


def xticks(df):
    return df["channels"].values, [f"{int(c)}" for c in df["channels"].values]


def plot_runtime(d):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for label, df in d.items():
        ax.plot(df["channels"], df["avg_ms"], "-o", ms=4, color=COLORS[label], label=label)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("channels $C$  (tokens $=2C$, layers $=C$, batch $=8$)")
    ax.set_ylabel("end-to-end runtime [ms]")
    ax.set_title("End-to-end runtime (Apple M1, log-log)")
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "end_to_end_baseline_vs_others_log.png"), dpi=180)
    plt.close(fig)


def plot_speedup(d, num, denom, fname, title):
    a = d[num].set_index("channels"); b = d[denom].set_index("channels")
    chans = sorted(set(a.index) & set(b.index))
    ratio = [b.loc[c, "avg_ms"] / a.loc[c, "avg_ms"] for c in chans]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(chans, ratio, "-o", ms=5, color="tab:green")
    for x, y in zip(chans, ratio):
        ax.annotate(f"{y:.0f}x" if y >= 10 else f"{y:.1f}x", (x, y),
                    textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("channels $C$  (tokens $=2C$, layers $=C$, batch $=8$)")
    ax.set_ylabel(f"speedup  ({denom} / {num} time)")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=180); plt.close(fig)




def plot_per_kernel():
    base = pd.read_csv(os.path.join(PK, PK_FILES["baseline"]))
    scal = pd.read_csv(os.path.join(PK, PK_FILES["scalar"]))
    simd = pd.read_csv(os.path.join(PK, PK_FILES["simd"]))
    big_t = int(base["tokens"].max())
    def at(df):
        x = df[df["tokens"] == big_t].set_index("kernel")["avg_ms"]
        return x
    b, s, v = at(base), at(scal), at(simd)
    kernels = ["equi_linear", "geometric_product", "equi_join",
               "equi_geometric_attention", "equi_rms_norm", "scaler_gated_gelu"]
    pretty = {"equi_linear": "EquiLinear", "geometric_product": "Geom.\nproduct",
              "equi_join": "Equi.\njoin", "equi_geometric_attention": "Attention",
              "equi_rms_norm": "RMS\nnorm", "scaler_gated_gelu": "Gated\nGELU"}
    scal_sp = [b[k] / s[k] for k in kernels]
    simd_sp = [b[k] / v[k] for k in kernels]
    x = np.arange(len(kernels)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.bar(x - w / 2, scal_sp, w, label="scalar (optimized)", color="tab:orange")
    ax.bar(x + w / 2, simd_sp, w, label="SIMD (NEON)", color="tab:green")
    for xi, val in zip(x - w / 2, scal_sp):
        ax.annotate(f"{val:.0f}x", (xi, val), textcoords="offset points", xytext=(0, 2), ha="center", fontsize=6.5)
    for xi, val in zip(x + w / 2, simd_sp):
        ax.annotate(f"{val:.0f}x", (xi, val), textcoords="offset points", xytext=(0, 2), ha="center", fontsize=6.5)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([pretty[k] for k in kernels], fontsize=8)
    ax.set_ylabel("speedup over baseline C++")
    ax.set_title(f"Per-kernel speedup at the largest size ($(B,T,C,L)$=$(8,128,64,64)$)")
    ax.grid(True, which="both", axis="y", ls=":", lw=0.4, alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "per_kernel_all_impls.png"), dpi=180); plt.close(fig)
    print("per-kernel speedups at T=%d:" % big_t)
    for k in kernels:
        print(f"  {k:<26} scalar {b[k]/s[k]:8.0f}x   simd {b[k]/v[k]:8.0f}x")


def main():
    d = load_e2e()
    plot_runtime(d)
    plot_speedup(d, "SIMD C++", "Baseline C++",
                 "baseline_cpp_vs_simd_cpp_end_to_end_speedup_ratio_log.png",
                 "End-to-end speedup: SIMD C++ over baseline")
    plot_speedup(d, "SIMD C++", "Python (EzGATr)",
                 "python_vs_simd_cpp_end_to_end_speedup_ratio_log.png",
                 "End-to-end speedup: SIMD C++ over Python")
    plot_per_kernel()
    a = d["SIMD C++"].set_index("channels"); bl = d["Baseline C++"].set_index("channels"); py = d["Python (EzGATr)"].set_index("channels")
    chans = sorted(set(a.index) & set(bl.index))
    print("\nend-to-end SIMD speedups:")
    print("  vs baseline: %.0fx .. %.0fx" % (min(bl.loc[c,'avg_ms']/a.loc[c,'avg_ms'] for c in chans),
                                             max(bl.loc[c,'avg_ms']/a.loc[c,'avg_ms'] for c in chans)))
    pc = sorted(set(a.index) & set(py.index))
    print("  vs python:   %.1fx .. %.1fx" % (min(py.loc[c,'avg_ms']/a.loc[c,'avg_ms'] for c in pc),
                                             max(py.loc[c,'avg_ms']/a.loc[c,'avg_ms'] for c in pc)))
    print("\nwrote 5 PNGs to", OUT)


if __name__ == "__main__":
    main()

"""
flop_byte_counter.py
====================
Analytical FLOPs and bytes-transferred estimates for each kernel in the
Optimized-C++ GATr implementation.

All counts are for a *
"FLOPs" means floating-point operations, each counted as 1 FLOP
(matching the convention used by most roofline literature).

Public API
----------
    from flop_byte_counter import count_kernel

    stats = count_kernel("equi_linear", B=1, T=64, C_in=8, C_out=8)
    print(stats["flops"], stats["bytes"])

Each returned dict has:
    flops       - total FLOPs (multiply + add each count as 1; FMAs as 2)
    bytes_read  - bytes loaded from memory
    bytes_write - bytes stored to memory
    bytes       - bytes_read + bytes_write
    intensity   - flops / bytes  (arithmetic intensity, FLOPs/byte)
    description - human-readable breakdown
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional


FLOAT_BYTES = 4   # float 32 throughout the implementation so 4 bytes per element


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _fma(n: int) -> int:
    """One multiply-accumulate = 2 FLOPs."""
    return 2 * n


# ---------------------------------------------------------------------------
# per Kernel counters
# ---------------------------------------------------------------------------

def count_geometric_product(*, B: int, T: int, C: int) -> dict:
    """
    geometric_product(x, y) -> z   shapes: (B, T, C, 16)

    The GP kernel (gp_kernel_one_batch) computes a 16×16 sparse
    multiplication table for PGA.  The table has 256 products of the form
      z[d] += sign * x[s1] * x[s2]
    From the PGA multiplication table the non-zero entries total ~112 FMAs
    (this is the standard PGA geometric-product sparsity; we use 112 as the
    conservative measured value; the exact count depends on grade filtering).
    We count 112 FMAs per multivector pair.
    """
    N = B * T * C          # number of multivector pairs
    MACS_PER_MV = 112      # non-zero entries in PGA GP table (both outputs)
    flops = _fma(N * MACS_PER_MV)

    # Read x and y; write z
    elem = B * T * C * 16
    bytes_r = 2 * elem * FLOAT_BYTES
    bytes_w =     elem * FLOAT_BYTES

    return _build(flops, bytes_r, bytes_w,
                  f"geometric_product B={B} T={T} C={C}")


def count_equi_dual(*, B: int, T: int, C: int) -> dict:
    """
    equi_dual(x) -> out   shapes: (..., 16)

    out[i] = DUAL_SIGN[i] * x[DUAL_PERM[i]]
    Pure permutation + sign flip: 1 multiply per blade → 16 FLOPs per MV.
    """
    N = B * T * C
    flops = N * 16          # 1 multiply per element, no additions

    elem = B * T * C * 16
    bytes_r = elem * FLOAT_BYTES
    bytes_w = elem * FLOAT_BYTES

    return _build(flops, bytes_r, bytes_w,
                  f"equi_dual B={B} T={T} C={C}")


def count_equi_join(*, B: int, T: int, C: int, has_reference: bool = True) -> dict:
    """
    equi_join(x, y, reference) -> out   shapes: (..., 16)

    Computes the join (regressive product) via dual + join_kernel + dual.
    The join kernel is a sparse bilinear form; non-zero entries ≈ 80 FMAs
    (by inspection of the generated join table for PGA).
    The optional reference broadcast is 16 multiplies per MV.
    """
    N = B * T * C
    MACS_JOIN = 80
    flops = _fma(N * MACS_JOIN)
    if has_reference:
        flops += N * 16     # scale by ref blade 14

    elem = B * T * C * 16
    bytes_r = 2 * elem * FLOAT_BYTES
    if has_reference:
        bytes_r += elem * FLOAT_BYTES   # reference tensor
    bytes_w = elem * FLOAT_BYTES

    return _build(flops, bytes_r, bytes_w,
                  f"equi_join B={B} T={T} C={C} ref={has_reference}")


def count_equi_linear(*, B: int, T: int, C_in: int, C_out: int) -> dict:
    """
    equi_linear(x, weight, basis) -> out
      x:      (B, T, C_in,  16)
      weight: (C_out, C_in, 9)
      out:    (B, T, C_out, 16)

    The kernel performs, per output token:
      For each (o, i): load 9 scaled weights, then accumulate into 16
      blade accumulators.  Each of the 9 weight slots contributes to at
      most 2 destination blades (diagonal + off-diagonal path).

    Counting from equi_linear_kernel:
      - 9 multiplies for weight scaling (wi[k] * Ck)              → 9
      - Diagonal slice (blades 0-15, covered by w0-w4):
          w0→1 blade, w1→4 blades, w2→6 blades, w3→4 blades, w4→1 blade
          = 16 FMAs
      - Off-diagonal (grade-lowering, w5-w8):
          w5→1, w6→3, w7→3, w8→1 = 8 FMAs
      Total per (o,i) pair = 9 + (16+8)*2 = 9 + 48 = 57 FLOPs
    """
    pairs = B * T * C_out * C_in
    FLOPS_PER_PAIR = 9 + _fma(16 + 8)   # 9 scale + 48 FMAs
    flops = pairs * FLOPS_PER_PAIR

    # Memory
    bytes_r = (
        B * T * C_in  * 16 * FLOAT_BYTES +   # x
        C_out * C_in  *  9 * FLOAT_BYTES      # weight
        # basis is a compile-time constant; kernel ignores the pointer
    )
    bytes_w = B * T * C_out * 16 * FLOAT_BYTES

    return _build(flops, bytes_r, bytes_w,
                  f"equi_linear B={B} T={T} C_in={C_in} C_out={C_out}")


def count_equi_rms_norm(*, B: int, T: int, C: int) -> dict:
    """
    equi_rms_norm(x, weight) -> out   shapes: (..., C, 16)

        Per token (N = B*T):
            - selected_inner_product_square: 8 squares + 7 adds = 15 FLOPs, × C channels
            - sum across channels: +1 add per channel
            - mean: 1 div
            - sqrt + reciprocal: ~2 FLOPs
            - scale + optional weight multiply: 1 mul per channel
            - apply scale: 16 mul per channel
    """
    N = B * T
    ip_flops   = N * C * (8 + 7 + 1)   # 8 squares, 7 adds, +1 sum add
    norm_flops = N * (1 + 2)           # mean div + sqrt + rcp
    scale_flops = N * C * (1 + 16)     # scale * weight, then apply to 16 blades
    flops = ip_flops + norm_flops + scale_flops

    elem = B * T * C * 16
    bytes_r = elem * FLOAT_BYTES + C * FLOAT_BYTES    # x + weight
    bytes_w = elem * FLOAT_BYTES

    return _build(flops, bytes_r, bytes_w,
                  f"equi_rms_norm B={B} T={T} C={C}")


def count_scaler_gated_gelu(*, B: int, T: int, C: int) -> dict:
    """
    scaler_gated_gelu(x) -> out   shapes: (..., C, 16)

    Per multivector (M = B*T*C):
      gate = 0.5*v*(1 + tanh(k0*(v + k1*v^3)))
           ≈ 1 cube + 2 mul + 1 add + 1 tanh + 1 add + 1 mul + 1 mul = ~8 FLOPs
      (tanh counts as 1 transcendental ≈ ~20 FLOPs on CPU, but we count it
       as 1 for the arithmetic-intensity lower bound.)
      out[k] = x[k] * gate  → 16 multiplies per MV
    """
    M = B * T * C
    GATE_FLOPS  = 8
    APPLY_FLOPS = 16
    flops = M * (GATE_FLOPS + APPLY_FLOPS)

    elem = B * T * C * 16
    bytes_r = elem * FLOAT_BYTES
    bytes_w = elem * FLOAT_BYTES

    return _build(flops, bytes_r, bytes_w,
                  f"scaler_gated_gelu B={B} T={T} C={C}")


def count_attention(
    *,
    B: int,
    H: int,
    Tq: int,
    Tk: int,
    Cqk: int,
    Cv: int,
) -> dict:
    """
    equi_geometric_attention (fused IPA + DAA SDPA)
      q_mv, k_mv: (B, H, Tq, Cqk, 16)
      v_flat:     (B, H, Tk,  Cv,  16)  -- Dv = Cv*16

    Phase 1 – trivector normalisation (fill_norm):
      per element (N = B*H*T*Cqk): 1 sq + 1 add + 1 div + 4 mul = 7 FLOPs
      run twice (q and k).

    Phase 2 – score computation (inner loop over i,j,c):
      IPA dot: 7 FMAs (blades 0,2,3,4,8,9,10) per channel
      DAA:
        qsum  = qn[0]^2+qn[1]^2+qn[2]^2                   3 sq + 2 add
        ksum  = same                                        3 sq + 2 add
        cross = dot(qn[:3], kn[:3])                        3 mul + 2 add
        daa   = -qsum*kn3^2 - qn3^2*ksum + 2*qn3*kn3*cross  7 FLOPs
        total per channel ≈ 22 FLOPs
      weight combine: 2 FMAs per channel (w_ipa*ipa + w_daa*daa)
      Total per (i,j,c): 14 + 22 + 4 = 40 FLOPs

    Phase 3 – softmax (per row of length Tk):
      exp × Tk + sum + div × Tk ≈ 3*Tk FLOPs

    Phase 4 – weighted sum of V (per query token):
      Tk × Dv FMAs  (Dv = Cv*16)
    """
    Dv = Cv * 16

    # Phase 1
    norm_flops = B * H * (Tq + Tk) * Cqk * 7

    # Phase 2
    score_flops = B * H * Tq * Tk * Cqk * 40

    # Phase 3
    softmax_flops = B * H * Tq * (3 * Tk)

    # Phase 4
    attn_flops = _fma(B * H * Tq * Tk * Dv)

    flops = norm_flops + score_flops + softmax_flops + attn_flops

    # Memory: read q, k, v; write out
    bytes_r = (
        B * H * Tq * Cqk * 16 * FLOAT_BYTES +   # q
        B * H * Tk * Cqk * 16 * FLOAT_BYTES +   # k
        B * H * Tk * Dv       * FLOAT_BYTES      # v
    )
    bytes_w = B * H * Tq * Dv * FLOAT_BYTES

    desc = (f"attention B={B} H={H} Tq={Tq} Tk={Tk} Cqk={Cqk} Cv={Cv} "
            f"[norm={norm_flops} score={score_flops} "
            f"softmax={softmax_flops} attn_v={attn_flops}]")
    return _build(flops, bytes_r, bytes_w, desc)


# ---------------------------------------------------------------------------
# Combined "full forward pass" counter (all kernels together)
# ---------------------------------------------------------------------------

def count_full_forward(
    *,
    B: int,
    T: int,
    C_in: int,
    num_layers: int,
) -> dict:
    """
    Approximate FLOP and byte count for one full forward pass of ASLMVOnlyGATrModel.
    The architecture perfectly reflects:
      - embedding: EquiLinear(C_in -> 32)
      - blocks (x num_layers):
          - mlp:
              - layer_norm(32)
              - proj_bil: EquiLinear(32 -> 128)
              - geometric_product(32) + equi_join(32)
              - proj_bil_out: EquiLinear(64 -> 32)
              - scaler_gated_gelu(32)
              - proj_out: EquiLinear(32 -> 32)
          - attn:
              - layer_norm(32)
              - proj_qkv: EquiLinear(32 -> 384)
              - attention(H=4, Cqk=32, Cv=32, Tq=T, Tk=T)
              - proj_out: EquiLinear(128 -> 32)
      - head: EquiLinear(32 -> 1)
    """
    hidden = 32
    H = 4
    
    parts = {}
    
    # 1. Embedding
    parts["embedding"] = count_equi_linear(B=B, T=T, C_in=C_in, C_out=hidden)
    
    # 2. Block loop
    block_total_flops = 0
    block_total_br = 0
    block_total_bw = 0
    
    # Track one block's costs mathematically
    b_parts = {
        # MLP
        "mlp_ln":       count_equi_rms_norm(B=B, T=T, C=hidden),
        "mlp_proj_bil": count_equi_linear(B=B, T=T, C_in=hidden, C_out=128),
        "mlp_gp":       count_geometric_product(B=B, T=T, C=32),
        "mlp_join":     count_equi_join(B=B, T=T, C=32),
        "mlp_bil_out":  count_equi_linear(B=B, T=T, C_in=64, C_out=hidden),
        "mlp_gelu":     count_scaler_gated_gelu(B=B, T=T, C=hidden),
        "mlp_proj_out": count_equi_linear(B=B, T=T, C_in=hidden, C_out=hidden),
        
        # Attention
        "attn_ln":      count_equi_rms_norm(B=B, T=T, C=hidden),
        "attn_qkv":     count_equi_linear(B=B, T=T, C_in=hidden, C_out=384),
        "attn_core":    count_attention(B=B, H=H, Tq=T, Tk=T, Cqk=32, Cv=32),
        "attn_out":     count_equi_linear(B=B, T=T, C_in=128, C_out=hidden),
    }
    
    for k, v in b_parts.items():
        block_total_flops += v["flops"]
        block_total_br += v["bytes_read"]
        block_total_bw += v["bytes_write"]
        parts[f"block_{k}"] = v  # Keep 1 block's stats for reference
        
    # Multiply by number of layers
    total_flops = parts["embedding"]["flops"] + num_layers * block_total_flops
    total_br = parts["embedding"]["bytes_read"] + num_layers * block_total_br
    total_bw = parts["embedding"]["bytes_write"] + num_layers * block_total_bw
    
    # 3. Head
    head_stats = count_equi_linear(B=B, T=T, C_in=hidden, C_out=1)
    parts["head"] = head_stats
    total_flops += head_stats["flops"]
    total_br += head_stats["bytes_read"]
    total_bw += head_stats["bytes_write"]

    result = _build(total_flops, total_br, total_bw,
                    f"full_forward B={B} T={T} C_in={C_in} Layers={num_layers}")
    result["parts"] = parts
    return result


# ---------------------------------------------------------------------------
# Dispatch helper
# ---------------------------------------------------------------------------

def count_kernel(name: str, **kwargs) -> dict:
    """
    Convenience dispatcher.

    Parameters for each kernel:
        geometric_product   B, T, C
        outer_product       B, T, C
        equi_dual           B, T, C
        equi_join           B, T, C, [has_reference=True]
        equi_linear         B, T, C_in, C_out
        equi_rms_norm       B, T, C
        scaler_gated_gelu   B, T, C
        attention           B, H, Tq, Tk, Cqk, Cv
        full_forward        B, T, C_in, num_layers
    """
    _map = {
        "geometric_product":  count_geometric_product,
        "equi_dual":          count_equi_dual,
        "equi_join":          count_equi_join,
        "equi_linear":        count_equi_linear,
        "equi_rms_norm":      count_equi_rms_norm,
        "scaler_gated_gelu":  count_scaler_gated_gelu,
        "attention":          count_attention,
        "full_forward":       count_full_forward,
    }
    if name not in _map:
        raise ValueError(f"Unknown kernel '{name}'. Choose from: {list(_map)}")
    return _map[name](**kwargs)


# ---------------------------------------------------------------------------
# Internal builder
# ---------------------------------------------------------------------------

def _build(flops: int, bytes_r: int, bytes_w: int, desc: str) -> dict:
    total_bytes = bytes_r + bytes_w
    intensity = flops / total_bytes if total_bytes > 0 else 0.0
    return {
        "flops":        flops,
        "bytes_read":   bytes_r,
        "bytes_write":  bytes_w,
        "bytes":        total_bytes,
        "intensity":    intensity,
        "description":  desc,
    }


# ---------------------------------------------------------------------------
# Quick self-test / pretty-print
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint

    configs = [
        ("geometric_product", dict(B=1, T=64, C=8)),
        ("equi_dual",         dict(B=1, T=64, C=8)),
        ("equi_join",         dict(B=1, T=64, C=8)),
        ("equi_linear",       dict(B=1, T=64, C_in=8, C_out=8)),
        ("equi_rms_norm",     dict(B=1, T=64, C=8)),
        ("scaler_gated_gelu", dict(B=1, T=64, C=8)),
        ("attention",         dict(B=1, H=4, Tq=64, Tk=64, Cqk=4, Cv=8)),
        ("full_forward",      dict(B=1, T=64, C_in=8, num_layers=4)),
    ]

    print(f"{'Kernel':<22} {'GFLOPs':>10} {'MB read':>10} {'MB write':>10} "
          f"{'MB total':>10} {'AI (F/B)':>10}")
    print("-" * 77)
    for name, kw in configs:
        s = count_kernel(name, **kw)
        print(f"{name:<22} "
              f"{s['flops']/1e9:>10.4f} "
              f"{s['bytes_read']/1e6:>10.3f} "
              f"{s['bytes_write']/1e6:>10.3f} "
              f"{s['bytes']/1e6:>10.3f} "
              f"{s['intensity']:>10.3f}")

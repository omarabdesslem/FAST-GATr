"""
Edit CONFIG below to set shapes, options, weights etc.

Each of the three C++ builds does different math so the flop counts differ.
The `build` selector gets passed down into every count function.

baseline = dense reference kernels (the 16^3 gp/join triple loop, the full
equi_linear basis contraction, reference concatenated-dot attention + the daa
einsum). optimized_for_cache = precomputed sparse generated gp/join kernels,
packed equi_linear, fused scalar attention. simd = NEON versions of the
generated gp/join kernels, packed equi_linear, fused NEON attention (padded to
a multiple of 4 channels, horizontal reduce per score).

Numbers here are meant to match the actual arithmetic in each build under
Code/cpp/.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, List


class Build(str, Enum):
    BASELINE = "baseline"
    OPTIMIZED = "optimized_for_cache"
    SIMD = "simd"

    @classmethod
    def parse(cls, text: str) -> "Build":
        key = text.strip().lower()
        # let people type a few shorthands
        aliases = {
            "opt": cls.OPTIMIZED,
            "optimized": cls.OPTIMIZED,
            "cache": cls.OPTIMIZED,
            "neon": cls.SIMD,
        }
        if key in aliases:
            return aliases[key]
        for b in cls:
            if b.value == key:
                return b
        raise ValueError(
            f"Unknown build '{text}'. Expected one of: "
            f"{', '.join(b.value for b in cls)}"
        )


@dataclass
class OpWeights:
    add: float = 1.0
    mul: float = 1.0
    div: float = 4.0
    sqrt: float = 8.0
    exp: float = 20.0
    tanh: float = 20.0
    pow2: float = 1.0


@dataclass
class FlopCount:
    add: float = 0.0
    mul: float = 0.0
    div: float = 0.0
    sqrt: float = 0.0
    exp: float = 0.0
    tanh: float = 0.0
    pow2: float = 0.0

    def __add__(self, other: "FlopCount") -> "FlopCount":
        return FlopCount(
            add=self.add + other.add,
            mul=self.mul + other.mul,
            div=self.div + other.div,
            sqrt=self.sqrt + other.sqrt,
            exp=self.exp + other.exp,
            tanh=self.tanh + other.tanh,
            pow2=self.pow2 + other.pow2,
        )

    def __iadd__(self, other: "FlopCount") -> "FlopCount":
        self.add += other.add
        self.mul += other.mul
        self.div += other.div
        self.sqrt += other.sqrt
        self.exp += other.exp
        self.tanh += other.tanh
        self.pow2 += other.pow2
        return self
    
    def scale(self, factor: float) -> "FlopCount":
        return FlopCount(
            add=self.add * factor,
            mul=self.mul * factor,
            div=self.div * factor,
            sqrt=self.sqrt * factor,
            exp=self.exp * factor,
            tanh=self.tanh * factor,
            pow2=self.pow2 * factor,
        )

    def flops_total(self) -> float:
        # every op counted as 1 flop
        return self.add + self.mul + self.div + self.sqrt + self.exp + self.tanh + self.pow2

    def weighted_flops(self, w: OpWeights) -> float:
        # ops cost different amounts depending on type
        return (
            self.add * w.add
            + self.mul * w.mul
            + self.div * w.div
            + self.sqrt * w.sqrt
            + self.exp * w.exp
            + self.tanh * w.tanh
            + self.pow2 * w.pow2
        )

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


# default op weights
WEIGHTS = OpWeights(
    add=1.0,
    mul=1.0,
    div=4.0,
    sqrt=8.0,
    exp=20.0,
    tanh=20.0,
    pow2=1.0,
)


@dataclass
class ScriptConfig:
    # which build's math to count - picks the kernels everywhere
    build: str = Build.BASELINE.value

    # Core shapes
    batch: int = 8
    tokens: int = 256
    heads: int = 1
    tk: int | None = None
    channels_in: int = 2
    channels_out: int = 32
    channels_qk: int = 2
    channels_v: int = 2

    # Attention options
    kinds: str = "ipa,daa"
    has_scalar: bool = False
    scalar_qk: int = 0
    scalar_v: int = 0
    has_mask: bool = False
    is_causal: bool = False
    custom_scale: bool = False
    value_nnz_frac: float = 1.0

    # EquiJoin options
    join_m: int | None = None
    join_reference: bool = True
    join_x_nnz_frac: float = 1.0
    include_join_kernel_build: bool = False
    noCompOpt: bool = True

    # RMS norm options
    norm_n: int | None = None
    norm_c: int | None = None
    norm_hasWt: bool = True
    norm_mul_by_one_free: bool = False

    # Standalone function sizes
    gp_m: int | None = None
    gelu_m: int | None = None

    # Operation weights
    w_add: float = WEIGHTS.add
    w_mul: float = WEIGHTS.mul
    w_div: float = WEIGHTS.div
    w_sqrt: float = WEIGHTS.sqrt
    w_exp: float = WEIGHTS.exp
    w_tanh: float = WEIGHTS.tanh
    w_pow2: float = WEIGHTS.pow2

    # JSON output path
    json_out: str | None = None

#setting config of all weights, options etc
CONFIG = ScriptConfig()

##########################
#dataclasses for diff modules/funcs
##########################

@dataclass
class EquiLinearCfg:
    batch: int
    tokens: int
    channels_in: int
    channels_out: int
    bias: bool = True
    build: Build = Build.BASELINE


@dataclass
class EquiRMSNormCfg:
    n: int
    channels: int
    hasWt: bool = True
    mOne: bool = False
    build: Build = Build.BASELINE


@dataclass
class GeometricProductCfg:
    m: int
    build: Build = Build.BASELINE


@dataclass
class EquiJoinCfg:
    m: int
    reference: bool = True
    x_nnz_frac: float = 1.0
    include_kernel_build: bool = False
    noCompOpt: bool = True #dont opt out sign mult
    build: Build = Build.BASELINE


@dataclass
class ScalerGatedGeluCfg:
    m: int
    build: Build = Build.BASELINE


@dataclass
class EquiGeometricAttentionCfg:
    batch: int
    heads: int
    t_query: int
    t_key: int
    channels_qk: int
    channels_v: int
    kinds: List[str]
    has_scalar: bool = False
    scalar_qk: int = 0
    scalar_v: int = 0
    has_mask: bool = False
    is_causal: bool = False
    custom_scale: bool = False
    value_nnz_frac: float = 1.0
    # which build's attention math to count.
    # BASELINE = the reference path, attention.cpp _raw_sdpa over a concatenated
    # q/k feature dim, plus the daa (4,4,5) einsum.
    # OPTIMIZED = fused scalar kernel _fused_sdpa_ipa_daa, with the daa algebra
    # collapsed and a per-channel fused dot over Cqk.
    # SIMD = fused NEON _fused_sdpa_ipa_daa, same algebra but channels padded to
    # round_up(Cqk, 4) and a horizontal reduce of the 4-lane accumulator per score.
    build: Build = Build.BASELINE

    @property
    def fused(self) -> bool:
        return self.build is not Build.BASELINE

    @property
    def simd_pad(self) -> bool:
        return self.build is Build.SIMD


#####################
# flop counts for each part seperately
#####################

def geometric_product_count(cfg: GeometricProductCfg) -> FlopCount:
    # baseline is the dense 16x16x16 triple loop over the bilinear basis,
    # sum += basis * x * y, so 2 mul + 1 add per term.
    if cfg.build is Build.BASELINE:
        terms = cfg.m * 16 * 16 * 16
        return FlopCount(mul=2 * terms, add=terms)
    # optimized = generated_kernels/gp_kernel.cpp, the hand-rolled sparse FMA
    # chain. just read the counts off it: 192 mul + 192 add per multivector.
    if cfg.build is Build.OPTIMIZED:
        return FlopCount(mul=192 * cfg.m, add=192 * cfg.m)
    # simd = gp_kernel_neon.cpp, same sparse map but vfmsq folds the negative
    # basis terms in, so 192 mul + 176 add per multivector.
    return FlopCount(mul=192 * cfg.m, add=176 * cfg.m)


def equi_linear_count(cfg: EquiLinearCfg) -> FlopCount:
    bt_co = cfg.batch * cfg.tokens * cfg.channels_out

    if cfg.build is Build.BASELINE:
        # baseline linear.cpp equi_linear_kernel does the full dense basis
        # contraction sum += wt * bs * xv over (d,i,w,s) = 16*Ci*9*16, which is
        # 2 mul + 1 add per term.
        terms = bt_co * 16 * cfg.channels_in * 9 * 16
        tally = FlopCount(mul=2 * terms, add=terms)
        if cfg.bias:
            # one bias add per output channel, outside the inner loop
            tally.add += bt_co
        return tally

    # optimized/simd use equi_linear_packed_kernel. The sparse basis wiring is
    # hard-coded and the basis norms get pre-folded into the packed weights, so
    # per (b,t,o) the inner channel loop is just 24 FMAs (1 mul + 1 add each)
    # over channels_in, plus 8 off-diagonal merge adds in the write-back.
    # Packing the weights is a one-time setup cost - not counted here, same as
    # how the C++ benchmark measures it.
    tally = FlopCount(
        mul=24 * cfg.channels_in * bt_co,
        add=24 * cfg.channels_in * bt_co + 8 * bt_co,
    )
    if cfg.bias:
        tally.add += bt_co
    return tally

def equi_rms_norm_count(cfg: EquiRMSNormCfg) -> FlopCount:
    # follows norm.cpp equi_rms_norm_kernel
    t = FlopCount()

    # ip += v*v, 8 times per channel
    inner_terms = cfg.n * cfg.channels * 8
    t.mul += inner_terms
    t.add += inner_terms

    # sum_inner += ip -> one add per channel
    t.add += cfg.n * cfg.channels

    # mean = sum_inner / C
    t.div += cfg.n

    # scale = 1 / sqrt(mean): one div, one sqrt
    t.sqrt += cfg.n
    t.div += cfg.n

    # s = scale * weight[c] (or * 1.0f when weight is null). The kernel always
    # does this mul whether or not hasWt is set.
    t.mul += cfg.n * cfg.channels

    # out[k] = x[k] * s, 16 per channel
    t.mul += cfg.n * cfg.channels * 16

    return t


def scaler_gated_gelu_count(cfg: ScalerGatedGeluCfg) -> FlopCount:
    # norm.cpp scaler_gated_gelu_kernel.
    # mul = 6 (gate) + 16 (output scale), plus 2 add and 1 tanh.
    return FlopCount(mul=22 * cfg.m, add=2 * cfg.m, tanh=cfg.m)


def _join_kernel_build_count(noCompOpt: bool) -> FlopCount:
    # dual.cpp _compute_efficient_join_kernel, gets called inside equi_join
    per_pair = FlopCount()
    if noCompOpt:
        per_pair.mul += 16 + 16 + 16

    # outer_product, 16*16*16 terms
    terms = 16 * 16 * 16
    per_pair.mul += 2 * terms
    per_pair.add += terms

    return per_pair.scale(16 * 16)


def equi_join_count(cfg: EquiJoinCfg) -> FlopCount:
    # dual.cpp equi_join: out[i] = sum_{j,k} kernel[i,j,k] * x[j] * y[k]
    if cfg.x_nnz_frac < 0.0 or cfg.x_nnz_frac > 1.0:
        raise ValueError(f"x_nnz_frac must be in [0, 1], got {cfg.x_nnz_frac}")

    if cfg.build is Build.BASELINE:
        # dense kernel-evaluation hot loop. The `if (xj == 0) continue` guard
        # is approximated by letting x_nnz_frac scale the effective j range.
        # 2 mul + 1 add per surviving term.
        effective_j = 16.0 * cfg.x_nnz_frac
        terms = cfg.m * 16.0 * effective_j * 16.0
        t = FlopCount(mul=2 * terms, add=terms)
        # baseline rebuilds the join kernel every call - optimized/simd use a
        # precomputed one, so only baseline pays this.
        if cfg.include_kernel_build:
            t += _join_kernel_build_count(cfg.noCompOpt)
    else:
        # optimized -> join_kernel.cpp, simd -> join_kernel_simd.cpp. Both run
        # the same precomputed sparse map at 81 mul + 81 add per multivector
        # (NEON is identical arithmetic, just 4 lanes at a time).
        t = FlopCount(mul=81 * cfg.m, add=81 * cfg.m)

    if cfg.reference:
        # out[i] *= reference[..., 14] -> 16 muls per multivector, every build
        t.mul += cfg.m * 16

    return t


def _daa_perEl_count() -> FlopCount:
    # reference daa transform (cpp/baseline compute_qk_for_daa, also the Python).
    # the einsum("ijk,...i,...j->...k", basis(4,4,5), ret, ret) is 4*4*5=80
    # terms at 2 mul + 1 add each -> 160 mul, 80 add. Then ret = tri*normalizer
    # over 4 blades (4 mul), and normalizer = e123/(e123^2+eps) which is 1 mul
    # for the square, 1 add, 1 div. Adds up to 165 mul, 81 add, 1 div.
    return FlopCount(mul=165, add=81, div=1)


def _kind_dim(kind: str) -> int:
    # how many features per channel each kind adds to the concatenated q/k dim
    # on the baseline reference path.
    #
    # ipa is 7: attention.cpp _ipa_blade_indices keeps the 7 inner-product
    # blades {0,2,3,4,8,9,10} (keep_tri_vector=False). Heads up - the upstream
    # ezgatr reference we validate against keeps 8 (it also keeps e_123). The
    # C++ drops e_123 and still passes because that blade barely matters for the
    # test inputs. We count what the C++ actually runs, so 7.
    # daa is 5: compute_qk_for_daa contracts the (4,4,5) basis down to 5.
    if kind == "ipa":
        return 7
    if kind == "daa":
        return 5
    raise ValueError(f"Unsupported attention kind '{kind}'. Expected one of: ipa, daa")


def _causal_kept(b: int, h: int, tq: int, tk: int, is_causal: bool) -> float:
    # how many (query, key) pairs survive causal masking, on average, for the
    # value accumulation. For square tq==tk it's the lower triangle, i.e.
    # (tk+1)/2 keys per query.
    if not is_causal:
        return float(b * h * tq * tk)
    if tq == tk:
        return float(b * h) * tq * (tk + 1) / 2.0
    # non-square case: just count the (i,j) with j<=i
    kept = sum(min(i + 1, tk) for i in range(tq))
    return float(b * h) * kept


def _equi_geometric_attention_reference(
    cfg: EquiGeometricAttentionCfg, has_ipa: bool, has_daa: bool
) -> FlopCount:
    # reference path - attention.cpp _raw_sdpa over a concatenated q/k feature
    # dim (and the Python impl). The daa q/k features come out of the full
    # (4,4,5) einsum, and the score is a plain dot over the concatenated
    # D = 7*Cqk (ipa) + 5*Cqk (daa).
    t = FlopCount()

    b = cfg.batch
    h = cfg.heads
    tq = cfg.t_query
    tk = cfg.t_key
    cqk = cfg.channels_qk
    cv = cfg.channels_v

    lq = b * h * tq * cqk
    lk = b * h * tk * cqk

    d_mv = 0
    for kind in (["ipa"] if has_ipa else []) + (["daa"] if has_daa else []):
        d_mv += cqk * _kind_dim(kind)
        if kind == "daa":
            # run the per-element daa transform on every q/k channel-token
            t += _daa_perEl_count().scale(lq + lk)
        # q_kind * weight[idx] across this kind's q features
        t.mul += b * h * tq * cqk * _kind_dim(kind)

    d = d_mv + (cfg.scalar_qk if cfg.has_scalar else 0)
    dv = (cv * 16) + (cfg.scalar_v if cfg.has_scalar else 0)
    if d <= 0:
        raise ValueError(f"Computed q/k feature dim D must be > 0, got {d}")
    if dv <= 0:
        raise ValueError(f"Computed value feature dim Dv must be > 0, got {dv}")

    # the score dot still runs over every (i,j) even with causal masking - the
    # kernel computes s for all j then sets j>i to -inf - so pos is the whole
    # grid. only the softmax and value accumulation drop down to kept.
    pos = float(b * h * tq * tk)
    kept = _causal_kept(b, h, tq, tk, cfg.is_causal)

    # dot = 0; for d: dot += q[d]*k[d] -> D mul + D add per (b,h,i,j) pair (the
    # += into the zeroed accumulator counts as a real add). then dot * scale.
    t.mul += pos * d
    t.add += pos * d
    t.mul += pos

    if cfg.has_mask:
        t.add += pos

    # softmax - exp/denom/normalize only touch the kept (unmasked) keys
    t.add += kept         # scores[j] - max_score
    t.exp += kept
    t.add += kept         # denom += e
    t.div += b * h * tq   # 1 / denom
    t.mul += kept         # scores[j] *= inv_denom

    # value accumulation o += a*v, over the kept keys
    accum_terms = kept * dv * cfg.value_nnz_frac
    t.mul += accum_terms
    t.add += accum_terms

    # default 1/sqrt(D) scale
    if not cfg.custom_scale:
        t.sqrt += 1
        t.div += 1

    return t


def equi_geometric_attention_count(cfg: EquiGeometricAttentionCfg) -> FlopCount:
    # picks the fused model (optimized/simd) or the reference one (baseline +
    # Python) off cfg.fused
    if cfg.value_nnz_frac < 0.0 or cfg.value_nnz_frac > 1.0:
        raise ValueError(f"value_nnz_frac must be in [0, 1], got {cfg.value_nnz_frac}")

    if not cfg.kinds:
        raise ValueError("kinds must be non-empty")

    norm_kinds = [k.strip().lower() for k in cfg.kinds]
    for kind in norm_kinds:
        if kind not in ("ipa", "daa"):
            raise ValueError(f"Unsupported attention kind '{kind}'. Expected one of: ipa, daa")
    has_ipa = "ipa" in norm_kinds
    has_daa = "daa" in norm_kinds

    if not cfg.fused:
        return _equi_geometric_attention_reference(cfg, has_ipa, has_daa)

    # fused kernel: attention.cpp _fused_sdpa_ipa_daa (optimized / simd).
    # the score for each (b,h,i,j) is computed straight from the multivectors -
    # the daa einsum is collapsed into a hand-derived algebraic form and the
    # per-(h,c) kind weights get folded in at FMA time. ends up much cheaper
    # than the 17*Cqk plain dot the reference would do.
    t = FlopCount()

    b = cfg.batch
    h = cfg.heads
    tq = cfg.t_query
    tk = cfg.t_key
    cqk = cfg.channels_qk
    cv = cfg.channels_v

    # channels actually walked by the score loop. simd pads to a multiple of 4
    # and runs FMAs over the zero lanes too; optimized walks exactly cqk.
    cscore = ((cqk + 3) // 4) * 4 if cfg.simd_pad else cqk

    # trivector normalization (pack / fill_norm), daa only.
    # per (token, channel): t3*t3 (1 mul), +eps (1 add), t3/(...) (1 div), then
    # tri[0..2]*inv and t3*inv (4 mul) = 5 mul, 1 add, 1 div. Walks the *real*
    # channels of every q/k token - no padding here.
    if has_daa:
        norm_tokens = b * h * (tq + tk) * cqk
        t.mul += 5 * norm_tokens
        t.add += 1 * norm_tokens
        t.div += 1 * norm_tokens

    pos = float(b * h * tq * tk)
    kept = _causal_kept(b, h, tq, tk, cfg.is_causal)

    # score channel-loop, per (b,h,i,j) pair, per channel walked.
    # the scalar (optimized) and NEON (simd) kernels do the same algebra but
    # accumulate differently, so the op splits aren't the same:
    #   scalar: ipa is a+b+...+g (6 adds for 7 muls); the daa combine
    #           -A*X - B*Y + 2*.. is 7 mul + 2 add; the per-kind fold
    #           dot += w_ipa*ipa + w_daa*daa is 2 mul + 2 add.
    #   simd:   ipa is vdup(0) + 7 vfmaq (7 mul + 7 add); the daa combine uses a
    #           vaddq for the doubling and two vfmsq -> 6 mul + 3 add; the fold
    #           is two vfmaq = 2 mul + 2 add.
    simd = cfg.simd_pad
    per_chan = FlopCount()
    if has_ipa:
        per_chan.mul += 7
        per_chan.add += 7 if simd else 6
    if has_daa:
        # qsum/ksum/cross are 3 mul + 2 add each -> 9 mul, 6 add (both builds)
        per_chan.mul += 9
        per_chan.add += 6
        # then the daa combine
        if simd:
            per_chan.mul += 6
            per_chan.add += 3
        else:
            per_chan.mul += 7
            per_chan.add += 2
    # fold each kind that's present into the running dot, 1 mul + 1 add apiece
    n_kinds = (1 if has_ipa else 0) + (1 if has_daa else 0)
    per_chan.mul += n_kinds
    per_chan.add += n_kinds

    t += per_chan.scale(pos * cscore)

    # per (b,h,i,j): simd has to horizontally reduce its 4-lane accumulator
    # (vaddvq, ~3 adds). the scalar kernel has no hsum since its per-channel
    # adds already chained the reduction. then dot * scale (1 mul) on both.
    if simd:
        t.add += 3 * pos
    t.mul += pos  # dot * scale

    if cfg.has_mask:
        t.add += pos

    # softmax - exp/denom/normalize only over the kept (unmasked) keys
    t.add += kept         # scores[j] - max_score
    t.exp += kept
    t.add += kept         # denom += e
    t.div += b * h * tq   # 1 / denom
    t.mul += kept         # scores[j] *= inv_denom

    # value accumulation o += a*v over Dv comps, kept keys only
    dv = (cv * 16) + (cfg.scalar_v if cfg.has_scalar else 0)
    if dv <= 0:
        raise ValueError(f"Computed value feature dim Dv must be > 0, got {dv}")
    accum_terms = kept * dv * cfg.value_nnz_frac
    t.mul += accum_terms
    t.add += accum_terms

    # default 1/sqrt(12*Cqk) scale - one sqrt + one div, done once
    if not cfg.custom_scale:
        t.sqrt += 1
        t.div += 1

    return t

####################
# helper functions
####################

def format(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x)):,}"
    return f"{x:,.3f}"


def print_report(name: str, tally: FlopCount, weights: OpWeights) -> None:
    print(f"\n{name}")
    print(f"  add  : {format(tally.add)}")
    print(f"  mul  : {format(tally.mul)}")
    print(f"  div  : {format(tally.div)}")
    print(f"  sqrt : {format(tally.sqrt)}")
    print(f"  exp  : {format(tally.exp)}")
    print(f"  tanh : {format(tally.tanh)}")
    print(f"  pow2 : {format(tally.pow2)}")
    print(f"  flops_total    : {format(tally.flops_total())}")
    print(f"  weighted_flops : {format(tally.weighted_flops(weights))}")


def _csv_kinds(text: str) -> List[str]:
    kinds = [k.strip().lower() for k in text.split(",") if k.strip()]

    seen = set()
    duplicates = set()
    for kind in kinds:
        if kind in seen:
            duplicates.add(kind)
        seen.add(kind)

    if duplicates:
        dup_text = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate in kinds not allowed: {dup_text}")

    return kinds

########################
# main function
########################

def main() -> None:
    cfg = CONFIG

    build = Build.parse(cfg.build)

    tk = cfg.tk if cfg.tk is not None else cfg.tokens
    gp_m = cfg.gp_m if cfg.gp_m is not None else (cfg.batch * cfg.tokens * cfg.channels_in)
    join_m = cfg.join_m if cfg.join_m is not None else gp_m
    gelu_m = cfg.gelu_m if cfg.gelu_m is not None else (cfg.batch * cfg.tokens * cfg.channels_out)
    norm_n = cfg.norm_n if cfg.norm_n is not None else (cfg.batch * cfg.tokens)
    norm_c = cfg.norm_c if cfg.norm_c is not None else cfg.channels_out

    # sanity-check the inputs
    for name, val in [
        ("batch", cfg.batch),
        ("tokens", cfg.tokens),
        ("heads", cfg.heads),
        ("tk", tk),
        ("channels_in", cfg.channels_in),
        ("channels_out", cfg.channels_out),
        ("channels_qk", cfg.channels_qk),
        ("channels_v", cfg.channels_v),
        ("gp_m", gp_m),
        ("join_m", join_m),
        ("gelu_m", gelu_m),
        ("norm_n", norm_n),
        ("norm_c", norm_c),
    ]:
        
        if val < 0:
            raise ValueError(f"{name} must be >= 0, got {val}")

    for name, val in [
        ("scalar_qk", cfg.scalar_qk),
        ("scalar_v", cfg.scalar_v),
        ("w_add", cfg.w_add),
        ("w_mul", cfg.w_mul),
        ("w_div", cfg.w_div),
        ("w_sqrt", cfg.w_sqrt),
        ("w_exp", cfg.w_exp),
        ("w_tanh", cfg.w_tanh),
        ("w_pow2", cfg.w_pow2),
    ]:
        if val < 0:
            raise ValueError(f"{name} must be >= 0, got {val}")

    if cfg.join_x_nnz_frac < 0.0 or cfg.join_x_nnz_frac > 1.0 or cfg.value_nnz_frac < 0.0 or cfg.value_nnz_frac > 1.0:
        raise ValueError(f"value_nnz_frac and join_x_nnz_frac must be in [0, 1], got {cfg.value_nnz_frac} and {cfg.join_x_nnz_frac}")

    if not cfg.has_scalar and (cfg.scalar_qk != 0 or cfg.scalar_v != 0):
        raise ValueError("CONFIG.scalar_qk/scalar_v are only used when CONFIG.has_scalar is True")

    kinds = _csv_kinds(cfg.kinds)
    if not kinds:
        raise ValueError("CONFIG.kinds must contain at least one kind (ipa and/or daa)")

    weights = OpWeights(
        add=cfg.w_add,
        mul=cfg.w_mul,
        div=cfg.w_div,
        sqrt=cfg.w_sqrt,
        exp=cfg.w_exp,
        tanh=cfg.w_tanh,
        pow2=cfg.w_pow2,
    )

    module_equi_linear_cfg = EquiLinearCfg(
        batch=cfg.batch,
        tokens=cfg.tokens,
        channels_in=cfg.channels_in,
        channels_out=cfg.channels_out,
        bias=True,
        build=build,
    )
    module_equi_rms_norm_cfg = EquiRMSNormCfg(
        n=norm_n,
        channels=norm_c,
        hasWt=cfg.norm_hasWt,
        mOne=cfg.norm_mul_by_one_free,
        build=build,
    )

    reports: Dict[str, FlopCount] = {
        "module::EquiLinear": equi_linear_count(module_equi_linear_cfg),
        "module::EquiRMSNorm": equi_rms_norm_count(module_equi_rms_norm_cfg),
        "functional::geometric_product": geometric_product_count(
            GeometricProductCfg(m=gp_m, build=build)
        ),
        "functional::equi_join": equi_join_count(
            EquiJoinCfg(
                m=join_m,
                reference=cfg.join_reference,
                x_nnz_frac=cfg.join_x_nnz_frac,
                include_kernel_build=cfg.include_join_kernel_build,
                noCompOpt=cfg.noCompOpt,
                build=build,
            )
        ),
        "functional::equi_geometric_attention": equi_geometric_attention_count(
            EquiGeometricAttentionCfg(
                batch=cfg.batch,
                heads=cfg.heads,
                t_query=cfg.tokens,
                t_key=tk,
                channels_qk=cfg.channels_qk,
                channels_v=cfg.channels_v,
                kinds=kinds,
                has_scalar=cfg.has_scalar,
                scalar_qk=cfg.scalar_qk,
                scalar_v=cfg.scalar_v,
                has_mask=cfg.has_mask,
                is_causal=cfg.is_causal,
                custom_scale=cfg.custom_scale,
                value_nnz_frac=cfg.value_nnz_frac,
                build=build,
            )
        ),
        "functional::scaler_gated_gelu": scaler_gated_gelu_count(
            ScalerGatedGeluCfg(m=gelu_m, build=build)
        ),
    }

    print(f"Weighted FLOP model for C++ kernels  [build: {build.value}]")
    print("weights:")
    print(f"  add={weights.add}, mul={weights.mul}, div={weights.div}, sqrt={weights.sqrt}, exp={weights.exp}, tanh={weights.tanh}, pow2={weights.pow2}")

    total = FlopCount()
    for name, tally in reports.items():
        print_report(name, tally, weights)
        total += tally

    print_report("TOTAL", total, weights)

    if cfg.json_out:
        payload = {
            "build": build.value,
            "weights": asdict(weights),
            "reports": {
                name: {
                    "ops": tally.as_dict(),
                    "flops_total": tally.flops_total(),
                    "weighted_flops": tally.weighted_flops(weights),
                }
                for name, tally in reports.items()
            },
            "total": {
                "ops": total.as_dict(),
                "flops_total": total.flops_total(),
                "weighted_flops": total.weighted_flops(weights),
            },
        }
        with open(cfg.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote JSON report: {cfg.json_out}")


if __name__ == "__main__":
    main()

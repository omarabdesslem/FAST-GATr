// attention.cpp
//
// Equivariant geometric attention for the GATr workload. Specialized for
// MV-only q/k/v with kinds = {ipa, daa}; other configurations are rejected
// at the dispatcher.


#include <torch/extension.h>
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "_cycle_counter.hpp"

#include <arm_neon.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

// add up the 4 lanes of a vector into one float
static inline float hsum4(float32x4_t v) {
    return vaddvq_f32(v);
}

namespace py = pybind11;
namespace ti = torch::indexing;

static torch::Tensor _tensor_from_py(py::object obj, const std::string& name) {
    PyObject* raw = obj.ptr();

    TORCH_CHECK(
        THPVariable_Check(raw),
        name, " must be a torch.Tensor"
    );

    return THPVariable_Unpack(raw);
}

/* ============================================================================
 * Shape helpers
 * ============================================================================ */

static torch::Tensor _flatten_ck(const torch::Tensor& mv) {
    // (..., C, K) -> (..., C*K)
    return torch::flatten(mv, -2, -1);
}

static torch::Tensor _inflate_ck(const torch::Tensor& mv, int64_t k = 16) {
    // (..., C*K) -> (..., C, K)
    auto sizes = mv.sizes().vec();

    TORCH_CHECK(
        !sizes.empty(),
        "_inflate_ck: input tensor must have at least one dimension"
    );

    TORCH_CHECK(
        sizes.back() % k == 0,
        "_inflate_ck: last dim ", sizes.back(),
        " is not divisible by k=", k
    );

    sizes.back() = sizes.back() / k;
    sizes.push_back(k);
    return mv.reshape(sizes);
}

/* ============================================================================
 * Geometric q/k transforms
 * ============================================================================ */

// Matches EzGATr _compute_inner_product_selector(..., keep_tri_vector=False).
// Excludes tri-vector blades 11, 12, 13, 14.
static torch::Tensor _ipa_blade_indices(torch::Device device) {
    return torch::tensor(
        {0, 2, 3, 4, 8, 9, 10},
        torch::TensorOptions().device(device).dtype(torch::kLong)
    );
}

static torch::Tensor _tri_vector_indices(torch::Device device) {
    return torch::tensor(
        {11, 12, 13, 14},
        torch::TensorOptions().device(device).dtype(torch::kLong)
    );
}

static std::tuple<torch::Tensor, torch::Tensor>
_daa_basis(torch::Device device, torch::ScalarType dtype) {
    auto bq = torch::zeros({4, 4, 5}, torch::dtype(dtype).device(device));
    auto bk = torch::zeros({4, 4, 5}, torch::dtype(dtype).device(device));

    auto r3 = torch::arange(
        3,
        torch::TensorOptions().device(device).dtype(torch::kLong)
    );

    bq.index_put_({r3, r3, 0}, 1.0);
    bk.index_put_({3, 3, 0}, -1.0);

    bq.index_put_({3, 3, 1}, 1.0);
    bk.index_put_({r3, r3, 1}, -1.0);

    bq.index_put_({r3, 3, r3 + 2}, 1.0);
    bk.index_put_({r3, 3, r3 + 2}, 2.0);

    return {bq, bk};
}

std::tuple<torch::Tensor, torch::Tensor>
compute_qk_for_ipa(torch::Tensor query, torch::Tensor key) {
    auto build_inner_vec = [](const torch::Tensor& q_or_k) {
        auto sel = _ipa_blade_indices(q_or_k.device());
        return torch::index_select(q_or_k, -1, sel);
    };

    return std::make_tuple(build_inner_vec(query), build_inner_vec(key));
}

std::tuple<torch::Tensor, torch::Tensor>
compute_qk_for_daa(torch::Tensor query, torch::Tensor key, double eps = 1e-3) {
    auto sel = _tri_vector_indices(query.device());

    auto basis = _daa_basis(query.device(), query.scalar_type());
    auto bq = std::get<0>(basis);
    auto bk = std::get<1>(basis);

    auto build = [&](const torch::Tensor& q_or_k, const torch::Tensor& basis_t) {
        auto tri = torch::index_select(q_or_k, -1, sel);  // (..., 4)

        auto e123 = tri.index({
            ti::Ellipsis,
            ti::Slice(3, 4)
        });  // (..., 1)

        auto normalizer = e123 / (e123.pow(2) + eps);
        auto ret = tri * normalizer;

        return torch::einsum("ijk,...i,...j->...k", {basis_t, ret, ret});
    };

    return {build(query, bq), build(key, bk)};
}


// Fused SDPA for the GATr-style MV-only setup with kinds = {ipa, daa}.
// Avoids the q_cat / k_cat materialization and the daa einsum: the dot is
// computed straight out of the MV tensors, with the per-(h, c) kind weights
// folded in at FMA time.
//
// Tensor shapes:
//   q_mv, k_mv  (B, H, T, Cqk, 16)  
//   v_flat      (B, H, T, Cv*16)    
//   w_ipa/w_daa (H, Cqk)           
static torch::Tensor _fused_sdpa_ipa_daa(
    const torch::Tensor& q_mv,
    const torch::Tensor& k_mv,
    const torch::Tensor& v_flat,
    const torch::Tensor& w_ipa,
    const torch::Tensor& w_daa,
    double daa_eps,
    const std::optional<torch::Tensor>& attn_mask,
    bool is_causal,
    std::optional<double> custom_scale
) {
    const int64_t B = q_mv.size(0);
    const int64_t H = q_mv.size(1);
    const int64_t Tq = q_mv.size(2);
    const int64_t Cqk = q_mv.size(3);
    const int64_t Tk = k_mv.size(2);
    const int64_t Dv = v_flat.size(3);

    // Default scale matches the Python path: feature dim is 12*Cqk (ipa)
    // + 5*Cqk (daa) = 17*Cqk.
    const float scale = custom_scale.has_value()
        ? static_cast<float>(*custom_scale)
        : 1.0f / std::sqrt(static_cast<float>(12 * Cqk));

    const float NEG_INF = -std::numeric_limits<float>::infinity();
    const float eps = static_cast<float>(daa_eps);

    const float* qp = q_mv.data_ptr<float>();
    const float* kp = k_mv.data_ptr<float>();
    const float* vp = v_flat.data_ptr<float>();
    const float* wip = w_ipa.data_ptr<float>();
    const float* wdp = w_daa.data_ptr<float>();

    // The dot product in the score loop reduces over the channel axis, but the
    // input is laid out blade-contiguous per channel, so a SIMD reduction over c
    // would need strided gathers every iteration. Transpose once here instead:
    // pull out the values we actually need and store them channel-contiguous, so
    // the inner loop is just aligned loads and FMAs.
    //
    // For ipa we keep the 7 inner-product blades; for daa we keep the normalized
    // trivector (4 comps). Pad the channel count up to a multiple of 4 and zero
    // the tail so the extra lanes drop out of the dot on their own.
    const int64_t Cpad = (Cqk + 3) & ~int64_t(3);
    static const int IPA_BLADES[7] = {0, 2, 3, 4, 8, 9, 10};

    auto pack = [&](const float* src, int64_t T,
                    std::vector<float>& ipa_out,
                    std::vector<float>& daa_out) {
        const int64_t NT = B * H * T;   // tokens
        ipa_out.assign(static_cast<size_t>(NT) * 7 * Cpad, 0.0f);
        daa_out.assign(static_cast<size_t>(NT) * 4 * Cpad, 0.0f);

        for (int64_t n = 0; n < NT; ++n) {
            const float* tok = src + n * Cqk * 16;   // this token's (Cqk, 16)
            float* ipa_tok = ipa_out.data() + n * 7 * Cpad;
            float* daa_tok = daa_out.data() + n * 4 * Cpad;

            for (int64_t c = 0; c < Cqk; ++c) {
                const float* mv = tok + c * 16;

                for (int bl = 0; bl < 7; ++bl) {
                    ipa_tok[bl * Cpad + c] = mv[IPA_BLADES[bl]];
                }

                // normalize the trivector (blades 11..14), same as the scalar path
                const float* tri = mv + 11;
                const float t3 = tri[3];
                const float inv = t3 / (t3 * t3 + eps);
                daa_tok[0 * Cpad + c] = tri[0] * inv;
                daa_tok[1 * Cpad + c] = tri[1] * inv;
                daa_tok[2 * Cpad + c] = tri[2] * inv;
                daa_tok[3 * Cpad + c] = t3     * inv;
            }
        }
    };

    std::vector<float> q_ipa, q_daa, k_ipa, k_daa;
    pack(qp, Tq, q_ipa, q_daa);
    pack(kp, Tk, k_ipa, k_daa);

    // Pad the weights the same way so they line up with the packed channels;
    // the zeroed tail keeps the padding lanes from contributing.
    std::vector<float> w_ipa_pad(static_cast<size_t>(H) * Cpad, 0.0f);
    std::vector<float> w_daa_pad(static_cast<size_t>(H) * Cpad, 0.0f);
    for (int64_t h = 0; h < H; ++h) {
        std::memcpy(w_ipa_pad.data() + h * Cpad, wip + h * Cqk, sizeof(float) * Cqk);
        std::memcpy(w_daa_pad.data() + h * Cpad, wdp + h * Cqk, sizeof(float) * Cqk);
    }

    std::vector<float> mask_buf;
    const bool have_mask = attn_mask.has_value();
    if (have_mask) {
        torch::Tensor m = *attn_mask;
        torch::Tensor mf;
        if (m.scalar_type() == torch::kBool) {
            mf = torch::where(
                m,
                torch::zeros({}, torch::dtype(torch::kFloat32).device(m.device())),
                torch::full({}, NEG_INF, torch::dtype(torch::kFloat32).device(m.device()))
            );
        } else {
            mf = m.to(torch::kFloat32);
        }
        mf = mf.expand({B, H, Tq, Tk}).contiguous().to(torch::kCPU);
        mask_buf.resize(static_cast<size_t>(B * H * Tq * Tk));
        std::memcpy(mask_buf.data(), mf.data_ptr<float>(), sizeof(float) * mask_buf.size());
    }

    auto out = torch::zeros({B, H, Tq, Dv}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    float* op = out.data_ptr<float>();

    std::vector<float> scores(static_cast<size_t>(Tk));

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            // Channel-major weight rows for this head (padded length Cpad).
            const float* w_ipa_h = w_ipa_pad.data() + h * Cpad;
            const float* w_daa_h = w_daa_pad.data() + h * Cpad;

            for (int64_t i = 0; i < Tq; ++i) {
                // packed rows for query token i (channel-contiguous)
                const int64_t qtok = (b * H + h) * Tq + i;
                const float* qi_ipa = q_ipa.data() + qtok * 7 * Cpad;
                const float* qi_daa = q_daa.data() + qtok * 4 * Cpad;

                float max_score = NEG_INF;

                for (int64_t j = 0; j < Tk; ++j) {
                    const int64_t ktok = (b * H + h) * Tk + j;
                    const float* kj_ipa = k_ipa.data() + ktok * 7 * Cpad;
                    const float* kj_daa = k_daa.data() + ktok * 4 * Cpad;

                    // running weighted dot, 4 channels per lane
                    float32x4_t acc = vdupq_n_f32(0.0f);

                    for (int64_t c = 0; c < Cpad; c += 4) {
                        // ipa: plain dot over the 7 kept blades
                        float32x4_t ipa = vdupq_n_f32(0.0f);
                        for (int bl = 0; bl < 7; ++bl) {
                            float32x4_t qv = vld1q_f32(qi_ipa + bl * Cpad + c);
                            float32x4_t kv = vld1q_f32(kj_ipa + bl * Cpad + c);
                            ipa = vfmaq_f32(ipa, qv, kv);
                        }

                        // daa: the (4,4,5) contraction collapses to a handful of
                        // dot products on the trivector. q3/k3 are the e123 comps.
                        float32x4_t q0 = vld1q_f32(qi_daa + 0 * Cpad + c);
                        float32x4_t q1 = vld1q_f32(qi_daa + 1 * Cpad + c);
                        float32x4_t q2 = vld1q_f32(qi_daa + 2 * Cpad + c);
                        float32x4_t q3 = vld1q_f32(qi_daa + 3 * Cpad + c);
                        float32x4_t k0 = vld1q_f32(kj_daa + 0 * Cpad + c);
                        float32x4_t k1 = vld1q_f32(kj_daa + 1 * Cpad + c);
                        float32x4_t k2 = vld1q_f32(kj_daa + 2 * Cpad + c);
                        float32x4_t k3 = vld1q_f32(kj_daa + 3 * Cpad + c);

                        float32x4_t qsum = vmulq_f32(q0, q0);   // |q_xyz|^2
                        qsum = vfmaq_f32(qsum, q1, q1);
                        qsum = vfmaq_f32(qsum, q2, q2);
                        float32x4_t ksum = vmulq_f32(k0, k0);   // |k_xyz|^2
                        ksum = vfmaq_f32(ksum, k1, k1);
                        ksum = vfmaq_f32(ksum, k2, k2);
                        float32x4_t cross = vmulq_f32(q0, k0);  // q_xyz . k_xyz
                        cross = vfmaq_f32(cross, q1, k1);
                        cross = vfmaq_f32(cross, q2, k2);

                        // daa = 2*(q3*k3)*cross - qsum*k3^2 - q3^2*ksum
                        float32x4_t daa = vmulq_f32(q3, k3);
                        daa = vmulq_f32(daa, cross);
                        daa = vaddq_f32(daa, daa);
                        daa = vfmsq_f32(daa, qsum, vmulq_f32(k3, k3));
                        daa = vfmsq_f32(daa, ksum, vmulq_f32(q3, q3));

                        // weight each kind per channel and fold into the dot
                        acc = vfmaq_f32(acc, vld1q_f32(w_ipa_h + c), ipa);
                        acc = vfmaq_f32(acc, vld1q_f32(w_daa_h + c), daa);
                    }

                    float dot = hsum4(acc);

                    float s = dot * scale;
                    if (have_mask) s += mask_buf[((b * H + h) * Tq + i) * Tk + j];
                    if (is_causal && j > i) s = NEG_INF;

                    scores[static_cast<size_t>(j)] = s;
                    if (s > max_score) max_score = s;
                }

                float* o_row = op + ((b * H + h) * Tq + i) * Dv;

                if (max_score == NEG_INF) {
                    for (int64_t d = 0; d < Dv; ++d) o_row[d] = 0.0f;
                    continue;
                }

                float denom = 0.0f;
                for (int64_t j = 0; j < Tk; ++j) {
                    const float e = std::exp(scores[static_cast<size_t>(j)] - max_score);
                    scores[static_cast<size_t>(j)] = e;
                    denom += e;
                }
                const float inv_denom = 1.0f / denom;
                for (int64_t j = 0; j < Tk; ++j) {
                    scores[static_cast<size_t>(j)] *= inv_denom;
                }

                for (int64_t d = 0; d < Dv; ++d) o_row[d] = 0.0f;
                // weighted sum of the value rows: o_row += softmax[j] * v[j].
                // Dv is a multiple of 16, so handle 4 vectors per step and let
                // the 4-wide / scalar tails mop up anything left over.
                for (int64_t j = 0; j < Tk; ++j) {
                    const float a = scores[static_cast<size_t>(j)];
                    if (a == 0.0f) continue;
                    const float32x4_t av = vdupq_n_f32(a);
                    const float* v_row = vp + ((b * H + h) * Tk + j) * Dv;

                    int64_t d = 0;
                    for (; d + 16 <= Dv; d += 16) {
                        float32x4_t o0 = vld1q_f32(o_row + d + 0);
                        float32x4_t o1 = vld1q_f32(o_row + d + 4);
                        float32x4_t o2 = vld1q_f32(o_row + d + 8);
                        float32x4_t o3 = vld1q_f32(o_row + d + 12);
                        o0 = vfmaq_f32(o0, av, vld1q_f32(v_row + d + 0));
                        o1 = vfmaq_f32(o1, av, vld1q_f32(v_row + d + 4));
                        o2 = vfmaq_f32(o2, av, vld1q_f32(v_row + d + 8));
                        o3 = vfmaq_f32(o3, av, vld1q_f32(v_row + d + 12));
                        vst1q_f32(o_row + d + 0,  o0);
                        vst1q_f32(o_row + d + 4,  o1);
                        vst1q_f32(o_row + d + 8,  o2);
                        vst1q_f32(o_row + d + 12, o3);
                    }
                    for (; d + 4 <= Dv; d += 4) {
                        float32x4_t ov = vld1q_f32(o_row + d);
                        ov = vfmaq_f32(ov, av, vld1q_f32(v_row + d));
                        vst1q_f32(o_row + d, ov);
                    }
                    for (; d < Dv; ++d) o_row[d] += a * v_row[d];   // leftover
                }
            }
        }
    }

    return out;
}

/* ============================================================================
 * Kind parsing
 * ============================================================================ */

using KindConfig = std::optional<std::map<std::string, double>>;
using KindsVec = std::vector<std::pair<std::string, KindConfig>>;

static KindsVec _parse_kinds(py::object kinds_obj) {
    py::dict kinds = py::reinterpret_borrow<py::dict>(kinds_obj);

    KindsVec out;
    out.reserve(kinds.size());

    for (auto item : kinds) {
        std::string name = py::cast<std::string>(item.first);

        if (item.second.is_none()) {
            out.push_back({name, std::nullopt});
        } else {
            std::map<std::string, double> cfg;
            py::dict cfg_dict = py::reinterpret_borrow<py::dict>(item.second);

            for (auto kv : cfg_dict) {
                cfg[py::cast<std::string>(kv.first)] =
                    py::cast<double>(kv.second);
            }

            out.push_back({name, cfg});
        }
    }

    return out;
}

static std::optional<std::vector<torch::Tensor>>
_parse_weight(py::object obj, const torch::Tensor& q, size_t num_kinds) {
    auto opts = torch::TensorOptions()
                    .dtype(q.scalar_type())
                    .device(q.device());

    auto make_defaults = [&]() {
        std::vector<torch::Tensor> defaults;
        defaults.reserve(num_kinds);

        for (size_t i = 0; i < num_kinds; ++i) {
            defaults.push_back(torch::ones({}, opts));
        }

        return defaults;
    };

    if (obj.is_none()) {
        return make_defaults();
    }

    std::vector<torch::Tensor> out;

    // Single tensor weight.
    if (THPVariable_Check(obj.ptr())) {
        out.push_back(THPVariable_Unpack(obj.ptr()));
        return out;
    }

    // Single numeric weight.
    try {
        double scalar = py::cast<double>(obj);
        out.push_back(torch::tensor(scalar, opts));
        return out;
    } catch (const py::cast_error&) {
        // Not numeric; treat as iterable below.
    }

    // Iterable/list/tuple of tensor or numeric weights.
    for (auto item : obj) {
        py::object pi = py::reinterpret_borrow<py::object>(item);

        if (THPVariable_Check(pi.ptr())) {
            out.push_back(THPVariable_Unpack(pi.ptr()));
            continue;
        }

        try {
            double scalar = py::cast<double>(pi);
            out.push_back(torch::tensor(scalar, opts));
            continue;
        } catch (const py::cast_error&) {
            TORCH_CHECK(
                false,
                "weight entries must be torch.Tensor or numeric scalar"
            );
        }
    }

    if (out.empty()) {
        return make_defaults();
    }

    return out;
}

/* ============================================================================
 * Q/K/V object parsing
 * ============================================================================ */

struct ParsedQKV {
    torch::Tensor q_mv;
    torch::Tensor k_mv;
    torch::Tensor v_mv;

    std::optional<torch::Tensor> q_scl;
    std::optional<torch::Tensor> k_scl;
    std::optional<torch::Tensor> v_scl;

    bool has_scalar = false;
};

static bool _is_tuple_like(py::object obj) {
    return py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj);
}

static std::pair<torch::Tensor, std::optional<torch::Tensor>>
_parse_one_qkv_object(py::object obj, const std::string& name) {
    if (_is_tuple_like(obj)) {
        py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);

        TORCH_CHECK(
            py::len(seq) == 2,
            name, " tuple/list must have length 2: (multi_vector, scalar)"
        );

        py::object mv_obj = py::reinterpret_borrow<py::object>(seq[0]);
        py::object scl_obj = py::reinterpret_borrow<py::object>(seq[1]);

        auto mv = _tensor_from_py(mv_obj, name + "[0]");
        auto scl = _tensor_from_py(scl_obj, name + "[1]");

        return {mv, scl};
    }

    return {_tensor_from_py(obj, name), std::nullopt};
}

static ParsedQKV _parse_qkv_objects(
    py::object query_obj,
    py::object key_obj,
    py::object value_obj
) {
    auto q = _parse_one_qkv_object(query_obj, "query");
    auto k = _parse_one_qkv_object(key_obj, "key");
    auto v = _parse_one_qkv_object(value_obj, "value");

    const bool q_has = q.second.has_value();
    const bool k_has = k.second.has_value();
    const bool v_has = v.second.has_value();

    TORCH_CHECK(
        q_has == k_has && q_has == v_has,
        "query, key, and value must either all be plain tensors or all be "
        "tuples/lists of the form (multi_vector, scalar)"
    );

    ParsedQKV parsed;
    parsed.q_mv = q.first;
    parsed.k_mv = k.first;
    parsed.v_mv = v.first;

    parsed.q_scl = q.second;
    parsed.k_scl = k.second;
    parsed.v_scl = v.second;

    parsed.has_scalar = q_has;

    return parsed;
}

/* ============================================================================
 * Main implementation
 * ============================================================================ */

static std::pair<torch::Tensor, std::optional<torch::Tensor>>
equi_geometric_attention_impl(
    const torch::Tensor& query_mv,                         // (B, H, T, Cqk, 16)
    const torch::Tensor& key_mv,                           // (B, H, T, Cqk, 16)
    const torch::Tensor& value_mv,                         // (B, H, T, Cv, 16)
    const std::optional<torch::Tensor>& query_scl,          // optional (B,H,T,Sqk)
    const std::optional<torch::Tensor>& key_scl,            // optional (B,H,T,Sqk)
    const std::optional<torch::Tensor>& value_scl,          // optional (B,H,T,Sv)
    const KindsVec& kinds,
    const std::vector<torch::Tensor>& weight,
    const std::optional<torch::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale
) {
    TORCH_CHECK(
        dropout_p == 0.0,
        "equi_geometric_attention: dropout_p must be 0 in this raw C-loop implementation"
    );

    CYCLE_SCOPE("equi_geometric_attention");

    TORCH_CHECK(query_mv.dim() == 5, "query MV must have shape (B, H, T, C, 16)");
    TORCH_CHECK(key_mv.dim() == 5, "key MV must have shape (B, H, T, C, 16)");
    TORCH_CHECK(value_mv.dim() == 5, "value MV must have shape (B, H, T, C, 16)");

    TORCH_CHECK(query_mv.size(-1) == 16, "query MV last dim must be 16");
    TORCH_CHECK(key_mv.size(-1) == 16, "key MV last dim must be 16");
    TORCH_CHECK(value_mv.size(-1) == 16, "value MV last dim must be 16");

    TORCH_CHECK(!kinds.empty(), "must specify at least one attention kind");

    TORCH_CHECK(
        weight.size() == kinds.size(),
        "weight length (", weight.size(),
        ") must match number of kinds (", kinds.size(), ")"
    );

    // This kernel is specialized for the GATr workload: MV-only q/k/v with
    // kinds = {ipa, daa}. Other configurations are not supported.
    TORCH_CHECK(
        !query_scl.has_value() && !key_scl.has_value() && !value_scl.has_value(),
        "equi_geometric_attention: scalar q/k/v inputs are not supported"
    );
    TORCH_CHECK(kinds.size() == 2, "equi_geometric_attention: expected exactly two kinds (ipa, daa)");

    int ipa_idx = -1;
    int daa_idx = -1;
    double daa_eps = 1e-3;

    for (size_t k = 0; k < kinds.size(); ++k) {
        if (kinds[k].first == "ipa") {
            ipa_idx = static_cast<int>(k);
        } else if (kinds[k].first == "daa") {
            daa_idx = static_cast<int>(k);
            if (kinds[k].second.has_value()) {
                auto it = kinds[k].second->find("eps");
                if (it != kinds[k].second->end()) daa_eps = it->second;
            }
        }
    }
    TORCH_CHECK(ipa_idx >= 0 && daa_idx >= 0,
        "equi_geometric_attention: kinds must contain both 'ipa' and 'daa'");

    const int64_t H = query_mv.size(1);
    const int64_t Cqk = query_mv.size(3);

    // attn_mix yields weights with shape (H, 1, Cqk, 1); the fused kernel
    // wants (H, Cqk). Anything else gets broadcast.
    auto to_hc = [&](const torch::Tensor& w) {
        auto wf = w.detach().contiguous().to(torch::kCPU, torch::kFloat32);
        if (wf.dim() != 0) wf = wf.squeeze();
        if (wf.dim() == 2 && wf.size(0) == H && wf.size(1) == Cqk) {
            return wf.contiguous();
        }
        if (wf.dim() == 1 && wf.size(0) == Cqk) {
            return wf.unsqueeze(0).expand({H, Cqk}).contiguous();
        }
        return wf.expand({H, Cqk}).contiguous();
    };

    auto w_ipa = to_hc(weight[static_cast<size_t>(ipa_idx)]);
    auto w_daa = to_hc(weight[static_cast<size_t>(daa_idx)]);

    auto q_c = query_mv.detach().contiguous().to(torch::kCPU, torch::kFloat32);
    auto k_c = key_mv  .detach().contiguous().to(torch::kCPU, torch::kFloat32);
    auto v_c = _flatten_ck(value_mv).detach().contiguous().to(torch::kCPU, torch::kFloat32);

    auto ret = _fused_sdpa_ipa_daa(
        q_c, k_c, v_c, w_ipa, w_daa, daa_eps,
        attn_mask, is_causal, scale
    );

    return {
        _inflate_ck(ret.to(query_mv.device(), query_mv.scalar_type()), 16),
        std::nullopt
    };
}

/* ============================================================================
 * Python bindings
 * ============================================================================ */

PYBIND11_MODULE(attention, m) {
    m.def(
        "compute_qk_for_ipa",
        &compute_qk_for_ipa,
        py::arg("query"),
        py::arg("key")
    );

    m.def(
        "compute_qk_for_daa",
        &compute_qk_for_daa,
        py::arg("query"),
        py::arg("key"),
        py::arg("eps") = 1e-3
    );

    m.def(
        "equi_geometric_attention",
        [](
            py::object query,
            py::object key,
            py::object value,
            py::object kinds,
            py::object weight,
            std::optional<torch::Tensor> attn_mask,
            double dropout_p,
            bool is_causal,
            std::optional<double> scale
        ) {
            ParsedQKV qkv = _parse_qkv_objects(query, key, value);

            auto parsed_kinds = _parse_kinds(kinds);
            auto parsed_weight_opt =
                _parse_weight(weight, qkv.q_mv, parsed_kinds.size());

            TORCH_CHECK(parsed_weight_opt.has_value(), "internal weight parse error");

            auto result = equi_geometric_attention_impl(
                qkv.q_mv,
                qkv.k_mv,
                qkv.v_mv,
                qkv.q_scl,
                qkv.k_scl,
                qkv.v_scl,
                parsed_kinds,
                *parsed_weight_opt,
                attn_mask,
                dropout_p,
                is_causal,
                scale
            );

            // Match upstream Python API: when there are no scalar features,
            // return just the MV tensor; only return a (mv, scl) tuple when
            // a scalar value tensor is actually present.
            if (result.second.has_value()) {
                return py::cast(std::make_tuple(result.first, *result.second));
            }
            return py::cast(std::make_tuple(result.first, py::none()));
        },
        py::arg("query"),
        py::arg("key"),
        py::arg("value"),
        py::arg("kinds"),
        py::arg("weight") = py::none(),
        py::arg("attn_mask") = std::nullopt,
        py::arg("dropout_p") = 0.0,
        py::arg("is_causal") = false,
        py::arg("scale") = std::nullopt
    );

    BIND_CYCLE_COUNTER(m);
}
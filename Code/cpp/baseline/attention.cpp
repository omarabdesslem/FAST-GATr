// attention.cpp
//
// Equivariant geometric attention matching ezgatr.nn.functional.attention.
// Supports both:
//   1. MV-only query/key/value tensors
//   2. tuple inputs: (multi-vector tensor, scalar tensor)
//
// Python reference behavior:
//   - If scalar channels are present, scalar q/k are prepended to the attention
//     feature dimension.
//   - Value is flattened MV value concatenated with scalar value.
//   - After SDPA, output is split back into (mv_output, scalar_output).
//
// SDPA hot path is implemented with raw C loops over float32 CPU buffers.
//Modified entire attention.cpp structure to support SDPA 


#include <torch/extension.h>
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "_cycle_counter.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

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

/* ============================================================================
 * Raw C-loop scaled dot-product attention
 * ============================================================================ */

static torch::Tensor _raw_sdpa(
    const torch::Tensor& q,                                    // (B, H, Tq, D)
    const torch::Tensor& k,                                    // (B, H, Tk, D)
    const torch::Tensor& v,                                    // (B, H, Tk, Dv)
    const std::optional<torch::Tensor>& attn_mask,
    bool is_causal,
    std::optional<double> custom_scale
) {
    TORCH_CHECK(q.dim() == 4, "raw SDPA: q must be 4D");
    TORCH_CHECK(k.dim() == 4, "raw SDPA: k must be 4D");
    TORCH_CHECK(v.dim() == 4, "raw SDPA: v must be 4D");

    TORCH_CHECK(q.size(0) == k.size(0), "raw SDPA: q/k batch mismatch");
    TORCH_CHECK(q.size(0) == v.size(0), "raw SDPA: q/v batch mismatch");

    TORCH_CHECK(q.size(1) == k.size(1), "raw SDPA: q/k head mismatch");
    TORCH_CHECK(q.size(1) == v.size(1), "raw SDPA: q/v head mismatch");

    TORCH_CHECK(k.size(2) == v.size(2), "raw SDPA: k/v sequence mismatch");
    TORCH_CHECK(q.size(3) == k.size(3), "raw SDPA: q/k feature mismatch");

    const auto out_device = q.device();
    const auto out_dtype = q.scalar_type();

    auto qc = q.detach().contiguous().to(torch::kCPU, torch::kFloat32);
    auto kc = k.detach().contiguous().to(torch::kCPU, torch::kFloat32);
    auto vc = v.detach().contiguous().to(torch::kCPU, torch::kFloat32);

    const float* qp = qc.data_ptr<float>();
    const float* kp = kc.data_ptr<float>();
    const float* vp = vc.data_ptr<float>();

    const int64_t B  = qc.size(0);
    const int64_t H  = qc.size(1);
    const int64_t Tq = qc.size(2);
    const int64_t Tk = kc.size(2);
    const int64_t D  = qc.size(3);
    const int64_t Dv = vc.size(3);

    const float scale = custom_scale.has_value()
        ? static_cast<float>(*custom_scale)
        : 1.0f / std::sqrt(static_cast<float>(D));

    const float NEG_INF = -std::numeric_limits<float>::infinity();

    std::vector<float> mask_buf;
    bool have_mask = attn_mask.has_value();

    if (have_mask) {
        torch::Tensor m = *attn_mask;
        torch::Tensor mf;

        if (m.scalar_type() == torch::kBool) {
            // PyTorch SDPA convention:
            // bool True means keep; False means masked out.
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
        std::memcpy(
            mask_buf.data(),
            mf.data_ptr<float>(),
            sizeof(float) * mask_buf.size()
        );
    }

    auto out = torch::zeros(
        {B, H, Tq, Dv},
        torch::dtype(torch::kFloat32).device(torch::kCPU)
    );

    float* op = out.data_ptr<float>();
    std::vector<float> scores(static_cast<size_t>(Tk));

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t i = 0; i < Tq; ++i) {
                const float* q_row = qp + ((b * H + h) * Tq + i) * D;

                float max_score = NEG_INF;

                for (int64_t j = 0; j < Tk; ++j) {
                    const float* k_row = kp + ((b * H + h) * Tk + j) * D;

                    float dot = 0.0f;
                    for (int64_t d = 0; d < D; ++d) {
                        dot += q_row[d] * k_row[d];
                    }

                    float s = dot * scale;

                    if (have_mask) {
                        s += mask_buf[((b * H + h) * Tq + i) * Tk + j];
                    }

                    if (is_causal && j > i) {
                        s = NEG_INF;
                    }

                    scores[static_cast<size_t>(j)] = s;

                    if (s > max_score) {
                        max_score = s;
                    }
                }

                float* o_row = op + ((b * H + h) * Tq + i) * Dv;

                if (max_score == NEG_INF) {
                    for (int64_t d = 0; d < Dv; ++d) {
                        o_row[d] = 0.0f;
                    }
                    continue;
                }

                float denom = 0.0f;

                for (int64_t j = 0; j < Tk; ++j) {
                    scores[static_cast<size_t>(j)] =
                        std::exp(scores[static_cast<size_t>(j)] - max_score);
                    denom += scores[static_cast<size_t>(j)];
                }

                const float inv_denom = 1.0f / denom;

                for (int64_t j = 0; j < Tk; ++j) {
                    scores[static_cast<size_t>(j)] *= inv_denom;
                }

                for (int64_t d = 0; d < Dv; ++d) {
                    o_row[d] = 0.0f;
                }

                for (int64_t j = 0; j < Tk; ++j) {
                    const float a = scores[static_cast<size_t>(j)];

                    if (a == 0.0f) {
                        continue;
                    }

                    const float* v_row = vp + ((b * H + h) * Tk + j) * Dv;

                    for (int64_t d = 0; d < Dv; ++d) {
                        o_row[d] += a * v_row[d];
                    }
                }
            }
        }
    }

    return out.to(out_device, out_dtype);
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

    const bool has_scalar = query_scl.has_value();

    TORCH_CHECK(
        has_scalar == key_scl.has_value() && has_scalar == value_scl.has_value(),
        "query_scl, key_scl, and value_scl must either all exist or all be None"
    );

    std::vector<torch::Tensor> qs;
    std::vector<torch::Tensor> ks;

    qs.reserve(kinds.size() + (has_scalar ? 1 : 0));
    ks.reserve(kinds.size() + (has_scalar ? 1 : 0));

    torch::Tensor value_for_sdpa;
    std::optional<int64_t> index_scl;

    if (has_scalar) {
        TORCH_CHECK(query_scl->dim() == 4, "query scalar must have shape (B,H,T,S)");
        TORCH_CHECK(key_scl->dim() == 4, "key scalar must have shape (B,H,T,S)");
        TORCH_CHECK(value_scl->dim() == 4, "value scalar must have shape (B,H,T,Sv)");

        // Python reference appends scalar q/k before MV q/k.
        qs.push_back(*query_scl);
        ks.push_back(*key_scl);

        value_for_sdpa = torch::cat({_flatten_ck(value_mv), *value_scl}, -1);
        index_scl = -value_scl->size(-1);
    } else {
        value_for_sdpa = _flatten_ck(value_mv);
        index_scl = std::nullopt;
    }

    for (size_t idx = 0; idx < kinds.size(); ++idx) {
        const std::string& kind = kinds[idx].first;
        const KindConfig& cfg = kinds[idx].second;

        torch::Tensor q_kind;
        torch::Tensor k_kind;

        if (kind == "ipa") {
            std::tie(q_kind, k_kind) = compute_qk_for_ipa(query_mv, key_mv);
        } else if (kind == "daa") {
            double eps = 1e-3;

            if (cfg.has_value()) {
                auto it = cfg->find("eps");
                if (it != cfg->end()) {
                    eps = it->second;
                }
            }

            std::tie(q_kind, k_kind) = compute_qk_for_daa(query_mv, key_mv, eps);
        } else {
            TORCH_CHECK(false, "Unknown attention kind: '", kind, "'");
        }

        // Python reference applies weight only to MV-derived q features.
        qs.push_back(_flatten_ck(q_kind * weight[idx]));
        ks.push_back(_flatten_ck(k_kind));
    }

    auto q_cat = torch::cat(qs, -1);
    auto k_cat = torch::cat(ks, -1);

    auto ret = _raw_sdpa(
        q_cat,
        k_cat,
        value_for_sdpa,
        attn_mask,
        is_causal,
        scale
    );

    if (index_scl.has_value()) {
        const int64_t scl_dim = value_scl->size(-1);
        const int64_t mv_end = ret.size(-1) - scl_dim;

        auto ret_mv = ret.index({
            ti::Ellipsis,
            ti::Slice(0, mv_end)
        });

        auto ret_scl = ret.index({
            ti::Ellipsis,
            ti::Slice(mv_end, ret.size(-1))
        });

        return {_inflate_ck(ret_mv, 16), ret_scl};
    }

    return {_inflate_ck(ret, 16), std::nullopt};
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
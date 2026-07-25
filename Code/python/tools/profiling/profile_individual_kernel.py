#!/usr/bin/env python3
"""
Supported:
geometric_product
outer_product
equi_dual
equi_join
asl_equi_linear
equi_geometric_attention
equi_rms_norm
"""

import time
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import torch
from einops import rearrange
from ezgatr.nets.mv_only_gatr import MVOnlyGATrConfig
from ezgatr_extensions.asl_mv_only_gatr import ASLMVOnlyGATrModel
from ops import linear, attention, norm, dual

def get_model_tensors():
    """Create a model and return intermediate tensors for kernel testing."""
    config = MVOnlyGATrConfig(size_channels_in=2)
    model = ASLMVOnlyGATrModel(config)
    model.eval()
    
    batch, tokens, channels = 4, 64, 2
    mv_dim = 16
    
    x = torch.randn(batch, tokens, channels, mv_dim)
    x_proj = torch.randn(batch, tokens, channels, mv_dim)
    
    return model, x, x_proj

def profile_full_model(num_iterations=100):
    """Profile the full model."""
    print(f"\nProfileing: Full Model (all kernels)")
    print(f"Iterations: {num_iterations}")
    print("-" * 60)
    
    model, x, _ = get_model_tensors()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    
    # Profile
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(num_iterations):
            output = model(x)
            if i % 20 == 0:
                _ = output.sum().item()
    elapsed = time.perf_counter() - start
    
    print(f"Total time: {elapsed:.2f}s")
    print(f"Avg per iteration: {elapsed*1000/num_iterations:.2f} ms")

def profile_geometric_product(num_iterations=100):
    """Profile only geometric_product"""
    print(f"\nProfileing: geometric_product")
    print(f"Iterations: {num_iterations}")
    print("-" * 60)
    
    config = MVOnlyGATrConfig(size_channels_in=2)
    model = ASLMVOnlyGATrModel(config)
    model.eval()
    
    batch, tokens, channels, mv_dim = 8, 512, 2, 16
    x = torch.randn(batch, tokens, channels, mv_dim)

    embedding = model.embedding(x)
    bil = model.blocks[0].mlp.equi_bil
    proj = bil.proj_bil(embedding)
    size_inter = config.size_channels_intermediate // 2
    
    # Slice dynamically based on what the tensor actually contains:
    chunks = torch.split(proj, size_inter, dim=-2)
    print(f"[DEBUG] proj shape: {proj.shape}, split chunks count: {len(chunks)}")
    
    # Safely assign the first two slices for geometric product profiling
    lg, rg = chunks[0], chunks[1] 


    geom_prod = linear.geometric_product
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = geom_prod(lg, rg)
    
    # Profile
    start = time.perf_counter()
    with torch.no_grad():
        
        with signposter.use_interval("Begin geometric_product loop", "End geometric_product loop"):
            for i in range(num_iterations):
                output = geom_prod(lg, rg)
                    
    elapsed = time.perf_counter() - start
    
    print(f"Total time: {elapsed:.2f}s")
    print(f"Avg per iteration: {elapsed*1000/num_iterations:.2f} ms")


def profile_equi_join(num_iterations=100):
    """Profile only equi_join"""
    print(f"\nProfileing: equi_join")
    print(f"Iterations: {num_iterations}")
    print("-" * 60)

    config = MVOnlyGATrConfig(size_channels_in=2)
    model = ASLMVOnlyGATrModel(config)
    model.eval()

    batch, tokens, channels, mv_dim = 8, 512, 2, 16
    x = torch.randn(batch, tokens, channels, mv_dim)

    embedding = model.embedding(x)
    bil = model.blocks[0].mlp.equi_bil
    proj = bil.proj_bil(embedding)
    size_inter = config.size_channels_intermediate // 2
    
    # Slice dynamically based on what the tensor actually contains:
    chunks = torch.split(proj, size_inter, dim=-2)
    print(f"[DEBUG] proj shape: {proj.shape}, split chunks count: {len(chunks)}")
    
    # Safely assign the first two slices for geometric product profiling
    lg, rg = chunks[0], chunks[1] 


    join_fn = dual.equi_join

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = join_fn(lj, rj)

    # Profile
    start = time.perf_counter()
    with torch.no_grad():
        with signposter.use_interval("Begin equi_join loop", "End equi_joinu loop"):
            for i in range(num_iterations):
                output = join_fn(lj, rj)
            
    elapsed = time.perf_counter() - start

    print(f"Total time: {elapsed:.2f}s")
    print(f"Avg per iteration: {elapsed*1000/num_iterations:.2f} ms")


def profile_equi_dual(num_iterations=100):
    """Profile only equi_dual"""
    print(f"\nProfileing: equi_dual")
    print(f"Iterations: {num_iterations}")
    print("-" * 60)

    batch, tokens, channels, mv_dim = 8, 512, 2, 16
    x = torch.randn(batch, tokens, channels, mv_dim)

    dual_fn = dual.equi_dual

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = dual_fn(x)

    # Profile
    start = time.perf_counter()
    with torch.no_grad():
        with signposter.use_interval("Begin equi_dual loop", "End equi_dual loop"):
            for i in range(num_iterations):
                output = dual_fn(x)
    elapsed = time.perf_counter() - start

    print(f"Total time: {elapsed:.2f}s")
    print(f"Avg per iteration: {elapsed*1000/num_iterations:.2f} ms")


def profile_asl_equi_linear(num_iterations=100):
    """Profile only ASL EquiLinear"""
    print(f"\nProfileing: asl_equi_linear")
    print(f"Iterations: {num_iterations}")
    print("-" * 60)

    config = MVOnlyGATrConfig(size_channels_in=2)
    model = ASLMVOnlyGATrModel(config)
    model.eval()

    batch, tokens, channels, mv_dim = 8, 512, 2, 16
    x = torch.randn(batch, tokens, channels, mv_dim)

    linear_layer = model.embedding

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = linear_layer(x)

    # Profile
    start = time.perf_counter()
    with torch.no_grad():
        with signposter.use_interval("Begin asl_equi_linear loop", "End asl_equi_linear loop"):
            for i in range(num_iterations):
                output = linear_layer(x)
                
    elapsed = time.perf_counter() - start

    print(f"Total time: {elapsed:.2f}s")
    print(f"Avg per iteration: {elapsed*1000/num_iterations:.2f} ms")


def profile_equi_geometric_attention(num_iterations=100):
    """Profile only equi_geometric_attention"""
    print(f"\nProfileing: equi_geometric_attention")
    print(f"Iterations: {num_iterations}")
    print("-" * 60)
    
    config = MVOnlyGATrConfig(size_channels_in=2)
    model = ASLMVOnlyGATrModel(config)
    model.eval()
    
    batch, tokens, channels, mv_dim = 8, 512, 2, 16
    x = torch.randn(batch, tokens, channels, mv_dim)

    embedding = model.embedding(x)
    mlp_out = model.blocks[0].mlp(embedding)
    attn_block = model.blocks[0].attn
    proj_qkv = attn_block.proj_qkv(mlp_out)
    q, k, v = rearrange(
        proj_qkv,
        "b t (qkv h c) k -> qkv b h t c k",
        qkv=3,
        h=config.attn_num_heads,
        c=config.size_channels_hidden,
    )

    basis = torch.randn(16, 16, 16)
    weight = [w.exp() for w in attn_block.attn_mix.values()]
    attn_fn = attention.equi_geometric_attention
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = attn_fn(
                q,
                k,
                v,
                kinds=attn_block.config.attn_kinds,
                weight=weight,
                attn_mask=None,
                is_causal=attn_block.config.attn_is_causal,
                dropout_p=attn_block.config.attn_dropout_p,
                scale=attn_block.config.attn_scale,
            )
    
    # Profile
    start = time.perf_counter()
    with torch.no_grad():
        with signposter.use_interval("Begin equi_geometric_attention loop", "End equi_geometric_attention loop"):
            for i in range(num_iterations):
                output = attn_fn(
                    q,
                    k,
                    v,
                    kinds=attn_block.config.attn_kinds,
                    weight=weight,
                    attn_mask=None,
                    is_causal=attn_block.config.attn_is_causal,
                    dropout_p=attn_block.config.attn_dropout_p,
                    scale=attn_block.config.attn_scale,
                )
            
    elapsed = time.perf_counter() - start
    
    print(f"Total time: {elapsed:.2f}s")
    print(f"Avg per iteration: {elapsed*1000/num_iterations:.2f} ms")

def profile_equi_rms_norm(num_iterations=100):
    """Profile only equi_rms_norm"""
    print(f"\nProfileing: equi_rms_norm")
    print(f"Iterations: {num_iterations}")
    print("-" * 60)
    
    config = MVOnlyGATrConfig(size_channels_in=2)
    model = ASLMVOnlyGATrModel(config)
    model.eval()
    
    batch, tokens, channels, mv_dim = 8, 512, 2, 16
    x = torch.randn(batch, tokens, config.size_channels_hidden, mv_dim)

    norm_fn = norm.asl_scaler_gated_gelu
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = norm_fn(x)
    
    # Profile
    start = time.perf_counter()
    with torch.no_grad():
        with signposter.use_interval("Begin equi_rms_norm loop", "End equi_rms_norm loop"):
            for i in range(num_iterations):
                output = norm_fn(x)
                
    elapsed = time.perf_counter() - start
    
    print(f"Total time: {elapsed:.2f}s")
    print(f"Avg per iteration: {elapsed*1000/num_iterations:.2f} ms")

def main():
    if len(sys.argv) < 2:
        kernel_name = "all"
        num_iters = 100
    else:
        kernel_name = sys.argv[1].lower()
        num_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    print("=" * 60)
    print("One by One Profiling with perf")
    print("=" * 60)
    print(f"Input: B=4, T=64, C=2, MV_DIM=16")
    
    if kernel_name == "all" or kernel_name == "full":
        profile_full_model(num_iters)
    elif kernel_name == "geometric_product":
        profile_geometric_product(num_iters)
    elif kernel_name == "equi_join":
        profile_equi_join(num_iters)
    elif kernel_name == "equi_dual":
        profile_equi_dual(num_iters)
    elif kernel_name == "asl_equi_linear":
        profile_asl_equi_linear(num_iters)
    elif kernel_name == "equi_geometric_attention" or kernel_name == "attention":
        profile_equi_geometric_attention(num_iters)
    elif kernel_name == "equi_rms_norm" or kernel_name == "norm":
        profile_equi_rms_norm(num_iters)
    else:
        print(f"\nUnknown: {kernel_name}")
        print("\nSupported cases:")
        print("all (or full)")
        print("geometric_product")
        print("equi_join")
        print("equi_dual")
        print("asl_equi_linear")
        print("equi_geometric_attention (or attention)")
        print("equi_rms_norm (or norm)")
        return 1
    
    print("\n" + "=" * 60)
    print("Profiling complete!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())

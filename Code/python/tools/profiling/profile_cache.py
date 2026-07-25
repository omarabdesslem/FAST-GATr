
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import torch
from ezgatr.nets.mv_only_gatr import MVOnlyGATrConfig
from ezgatr_extensions.asl_mv_only_gatr import ASLMVOnlyGATrModel

def profile_kernel( num_iterations=100):
    print(f"\n{'='*60}")
    print(f"Profiling end to end for cache misses etc.")
    print(f"{'='*60}")
    
    # Create config and model
    config = MVOnlyGATrConfig(size_channels_in=2)
    model = ASLMVOnlyGATrModel(config)
    model.eval()
    
    # Input shapes (batch, tokens, channels, dim)
    batch, tokens, channels = 8, 512, 2
    x = torch.randn(batch, tokens, channels, 16)
    
    # Warmup
    """ print(f"Warmup ({3} runs)...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(x) """
    
    # Profile
    print(f"Profiling ({num_iterations} iterations)...")
    # Example:
    print("")
    
    start = time.perf_counter()
    
    with torch.no_grad():
        with signposter.use_interval("Begin equi_geometric_attention loop", "End equi_geometric_attention loop"):
            for i in range(num_iterations):
                output = model(x)
    
    elapsed = time.perf_counter() - start
    
    print(f"\nCompleted in {elapsed:.2f}s")
    print(f"Avg per iteration: {elapsed*1000/num_iterations:.2f} ms")
    print(f"Total iterations: {num_iterations}")

if __name__ == "__main__":
    import sys
    
    num_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    
    print("="*60)
    print("Per-Kernel Profiling with perf")
    print("="*60)
    print(f"\nInput: B=4, T=32, C=2")
    print(f"Iterations: {num_iters}")

    profile_kernel( num_iterations=num_iters)
    
    print("\n" + "="*60)
    print("Profiling complete!")
    print("="*60)
    print("\nTo analyze results:")
    print("  1. Look at 'cache-misses' and 'L1-dcache-load-misses' above")
    print("  2. Higher cache-miss % indicates memory bandwidth bottleneck")
    print("  3. Run 'perf annotate' to see which lines have most misses")

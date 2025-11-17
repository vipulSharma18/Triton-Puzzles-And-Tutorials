"""
Correctness, autotuning, and benchmarking for fused softmax by myself.
Reference: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html#sphx-glr-getting-started-tutorials-02-fused-softmax-py
"""

import torch
import triton
import triton.language as tl


def torch_reference(x):
    pass

@triton.autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_SIZE': 128}),
    ],
    key=['numel'],
)
@triton.jit
def kernel(x_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pass

def triton_launcher(x):
    numel = x.numel()
    grid = lambda META: (triton.cdiv(numel,META['BLOCK_SIZE']),)
    kernel[grid](x, numel)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_args=['numel'],
        line_arg='provider',
        line_vals=['torch', 'triton'],
    )
)
def benchmark(numel, provider):
    pass

def main():
    torch.random.seed(0)
    print("Correctness and autotuning:")

    for size in [100, 1000]:
        x = torch.randn(size)
        output_torch = torch_reference(x)
        output_triton = triton_launcher(x)
        assert (output_torch-output_triton)==0, f'Mismatched outputs (torch, triton): {output_torch}, {output_triton}'

    benchmark.run(show_plots=True, save_path="./")


if __name__ == "__main__":
    main()
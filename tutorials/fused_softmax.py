"""
Correctness, autotuning, and benchmarking for fused softmax by myself.
Reference: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html#sphx-glr-getting-started-tutorials-02-fused-softmax-py
"""
import os

os.environ["TRITON_INTERPRET"] = "0"
os.environ["TRITON_DEBUG"] = "0"
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def python_reference(x):
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
        x_names=['numel'],
        line_arg='provider',
        line_vals=['torch', 'triton'],
    )
)
def benchmark(numel, provider):
    pass

def main():
    torch.random.seed(0)
    print("Correctness and autotuning:")

    y_sizes = [4096, 1823, 100]
    x_sizes = [128*i for i in range(2, 100)]
    for m in y_sizes:
        for n in x_sizes:
            x = torch.randn(m, n, device=DEVICE)
            output_naive = python_reference(x)
            output_torch = torch.softmax(x, axis=1)
            output_triton = triton_launcher(x)
            assert torch.allclose(output_naive, output_torch) and torch.allclose(output_torch, output_triton)
    print("Passed correctness. Running bechmark:")
    benchmark.run(print_data=True, save_path="./")


if __name__ == "__main__":
    main()
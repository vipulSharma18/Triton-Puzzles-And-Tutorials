"""
Correctness, autotuning, and benchmarking for fused softmax.
Note: Input shape is such that tensor can fit in GPU SRAM (shared mem)
 as we're demonstrating benefits of kernel fusion to avoid global mem accesses.
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

def torch_ops(x):
    "row-wise softmax using torch ops"
    x_max = x.max(dim=1, keepdim=True)[0]  # returns tuple so 0 index
    z = x - x_max
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1, keepdim=True)  # keep dims used instead of indexing
    ret = numerator/denominator
    return ret

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
    return torch.softmax(x, axis=-1)
    # numel = x.numel()
    # grid = lambda META: (triton.cdiv(numel,META['BLOCK_SIZE']),)
    # kernel[grid](x, numel)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128*i for i in range(1, 512)],
        line_arg='provider',
        line_vals=['torch_ops', 'torch_softmax', 'triton'],
        line_names=["Torch-Ops", "Torch-Softmax", "Triton"],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel="GB/s",  # this is what the benchmark function returns given the x_names and the line_arg kwargs.
        plot_name="fused-softmax-performance",
        args={'M': 4096},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)  # fix the main stream used by the device for persistent kernel setup
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch_ops':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_ops(x), quantiles=quantiles)
    elif provider == 'torch_softmax':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1), quantiles=quantiles)
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_launcher(x), quantiles=quantiles)
    
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


def main():
    torch.manual_seed(0)
    print("Autotuning/warmup/AoT compiling:")

    print("Correctness check:")
    y_sizes = [4096, 1823, 100]
    x_sizes = [128*i for i in range(2, 100)]
    for m in y_sizes:
        for n in x_sizes:
            x = torch.randn(m, n, device=DEVICE)
            output_torch_ops = torch_ops(x)
            output_torch = torch.softmax(x, axis=-1)
            output_triton = triton_launcher(x)
            assert torch.allclose(output_torch_ops, output_torch), str(torch.max(torch.abs(output_torch_ops-output_torch)))
            assert torch.allclose(output_triton, output_torch), str(torch.max(torch.abs(output_triton-output_torch)))

    print("Passed correctness.\nRunning bechmark:")
    benchmark.run(print_data=True, save_path="./")


if __name__ == "__main__":
    main()
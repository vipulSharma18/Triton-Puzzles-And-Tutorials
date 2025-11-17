"""
Basic vector addition  using Triton.
Reference: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#sphx-glr-getting-started-tutorials-01-vector-add-py
"""
import os

os.environ["TRITON_INTERPRET"] = "0"  # set to "1" for running on CPU, or the interpreter mode, and to enable pdb in triton kernels.
os.environ["TRITON_DEBUG"] = "0"  # set to "1" for device side asserts/debugging
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

import torch
import triton
import triton.language as tl


def torch_reference(x: torch.Tensor, y: torch.Tensor):
    """
    X and Y are 1D vectors with size ranging from 2**12 to 2**28.
    """
    return x + y


@triton.autotune(configs=[
        triton.Config(kwargs={"block_size": 2048}),
        triton.Config(kwargs={"block_size": 1024}),
        triton.Config(kwargs={"block_size": 512}),
        triton.Config(kwargs={"block_size": 256}),
    ],
    key=['num_elements'],
)
@triton.jit
def triton_vector_add(x_ptr: tl.tensor, y_ptr: tl.tensor, z_ptr: tl.tensor, num_elements: tl.int32, block_size: tl.constexpr):
    curr_block_start = tl.program_id(0)*block_size
    curr_range = tl.arange(0, block_size) + curr_block_start
    x = tl.load(x_ptr + curr_range, curr_range<num_elements)
    y = tl.load(y_ptr + curr_range, curr_range<num_elements)
    tl.store(z_ptr + curr_range, x+y, curr_range<num_elements)
    # can't have return None here, will cause compilation fail


def launch_vector_add_empty_like(x: torch.Tensor, y: torch.Tensor):
    z = torch.empty_like(x)  # empty like is probably faster than zeros like cause it can let the tensor have garbage vals
    num_elements = x.numel()
    # block_size = 1024
    grid = lambda launch_params: (
        triton.cdiv(num_elements, launch_params['block_size']),
    )
    # will autotune block size for each pid
    triton_vector_add[grid](x, y, z, num_elements)
    # profile to see impact of no cuda sync and kernel running async as we return z here.
    return z


def launch_vector_add_zeros_like(x: torch.Tensor, y: torch.Tensor):
    z = torch.zeros_like(x)
    num_elements = x.numel()
    # block_size = 1024
    grid = lambda launch_params: (
        triton.cdiv(num_elements, launch_params['block_size']),
    )
    # will autotune block size for each pid
    triton_vector_add[grid](x, y, z, num_elements)
    # profile to see impact of no cuda sync and kernel running async as we return z here.
    return z


def main():
    device = triton.runtime.driver.active.get_active_torch_device()
    torch.manual_seed(0)
    sizes = [2**i for i in range(12, 28, 1)]

    print("Correctness check and autotuning for triton vector add.")    
    for size in sizes:
        x = torch.rand(size, device=device)
        y = torch.rand(size, device=device)
        output_torch = torch_reference(x, y)
        output_triton_zeros = launch_vector_add_zeros_like(x, y)
        output_triton_empty = launch_vector_add_empty_like(x, y)

        x = x.cpu()
        y = y.cpu()
        output_torch = output_torch.cpu()
        output_triton_zeros = output_triton_zeros.cpu()
        output_triton_empty = output_triton_empty.cpu()

        diff1 = torch.max(torch.abs(output_torch - output_triton_zeros))
        diff2 = torch.max(torch.abs(output_torch - output_triton_empty))

        assert diff1==0 and diff2==0, f'The maximum difference between torch and triton is f{diff1}, f{diff2}.'

        del x, y, output_torch, output_triton_zeros, output_triton_empty
        torch.cuda.empty_cache()

    print("Benchmarking torch and triton kernels.")
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['num_elements'],  # Argument names to use as an x-axis for the plot.
            x_vals=sizes,  # Different possible values for `x_name`.
            x_log=True,  # x axis is logarithmic.
            line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
            line_vals=['triton_empty', 'triton_zero', 'torch'],  # Possible values for `line_arg`.
            line_names=['Triton-O/P Empty-Init', 'Triton-O/P Zero-Init', 'Torch'],  # Label name for the lines.
            styles=[('blue', '-'), ('red', '-'), ('green', '-')],  # Line styles.
            ylabel='GB/s',  # Label name for the y-axis.
            plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
            args={},  # Values for function arguments not in `x_names` and `y_name`.
        ))
    def benchmark(num_elements, provider):
        x = torch.rand(num_elements, device=device, dtype=torch.float32)
        y = torch.rand(num_elements, device=device, dtype=torch.float32)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_reference(x, y), quantiles=quantiles)
        if provider == 'triton_zero':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: launch_vector_add_zeros_like(x, y), quantiles=quantiles)
        if provider == 'triton_empty':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: launch_vector_add_empty_like(x, y), quantiles=quantiles)
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(print_data=True, save_path="./")


if __name__ == "__main__":
    main()

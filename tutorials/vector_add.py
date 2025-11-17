"""
Basic vector addition  using Triton.
Reference: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#sphx-glr-getting-started-tutorials-01-vector-add-py
"""
import torch
import triton
import triton.language as tl


def torch_reference(x: torch.Tensor, y: torch.Tensor):
    """
    X and Y are 1D vectors with size ranging from 2**12 to 2**28.
    """
    return x + y


@triton.jit
def triton_vector_add(x_ptr: tl.tensor, y_ptr: tl.tensor, z_ptr: tl.tensor, num_elements: tl.int32, block_size: tl.constexpr):
    curr_block_start = tl.program_id(0)*block_size
    curr_range = tl.arange(0, block_size) + curr_block_start
    x = tl.load(x_ptr + curr_range, curr_range<num_elements)
    y = tl.load(y_ptr + curr_range, curr_range<num_elements)
    tl.store(z_ptr + curr_range, x+y, curr_range<num_elements)
    # can't have return None here, will cause compilation fail

def launch_vector_add(x: torch.Tensor, y: torch.Tensor):
    z = torch.empty_like(x)  # empty like is probably faster than zeros like cause it can let the tensor have garbage vals
    num_elements = x.numel()
    block_size = 1024
    grid = lambda launch_params: (
        triton.cdiv(num_elements, launch_params['block_size']),
    )
    # will autotune block size for each pid
    triton_vector_add[grid](x, y, z, num_elements, block_size)
    # profile to see impact of no cuda sync and kernel running async as we return z here.
    return z


def main():
    print("Benchmarking triton vector add against torch op!")
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    output_torch = x + y
    output_triton = launch_vector_add(x, y)
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')

if __name__ == "__main__":
    main()

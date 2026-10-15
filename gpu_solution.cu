/**
 * The file in which you will implement your GPU solutions!
 */

#include "algorithm_choices.h"

#include <algorithm>
#include <chrono>   // for timing
#include <iostream> // std::cout, std::endl
#include <limits>

#include "cuda_common.h"

namespace csc485b {
namespace a1 {
namespace gpu {

/**
 * The CPU baseline benefits from warm caches because the data was generated on
 * the CPU. Run the data through the GPU once with some arbitrary logic to
 * ensure that the GPU cache is warm too and the comparison is more fair.
 */
__global__ void warm_the_gpu(element_t *data, std::size_t invert_at_pos,
                             std::size_t num_elements) {
  int const th_id = blockIdx.x * blockDim.x + threadIdx.x;

  // We know this will never be true, because of the data generator logic,
  // but I doubt that the compiler will figure it out. Thus every element
  // should be read, but none of them should be modified.
  if (th_id < num_elements && data[th_id] > num_elements * 100) {
    ++data[th_id]; // should not be possible.
  }
}

/**
 * Each individual bitonic swap.
 */
__global__ void bitonic_swap(element_t *data, std::size_t num_elements,
                             std::size_t stride, std::size_t selection) {
  std::size_t th_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (th_id >= num_elements) {
    return;
  }

  std::size_t partner = th_id ^ stride;

  if (partner > th_id && partner < num_elements) {
    bool ascending = ((th_id & selection) == 0);

    element_t a = data[th_id];
    element_t b = data[partner];

    if ((ascending && a > b) || (!ascending && a < b)) {
      data[th_id] = b;
      data[partner] = a;
    }
  }
}

/**
 * Invert the array starting from the given position.
 */
__global__ void opposing_sort(element_t *data, std::size_t invert_at_pos,
                              std::size_t num_elements) {
  int const th_id = blockIdx.x * blockDim.x + threadIdx.x;

  std::size_t tail_size = num_elements - invert_at_pos;

  if (th_id < tail_size / 2) {
    std::size_t left = invert_at_pos + th_id;
    std::size_t right = num_elements - 1 - th_id;

    element_t temp = data[left];
    data[left] = data[right];
    data[right] = temp;
  }
}

/**
 * Marking out where for the swaps to occur.
 */
__global__ void radix_scan(const element_t *data, unsigned int *prefix,
                           unsigned int *block_totals, std::size_t n, int bit) {
  __shared__ unsigned int shared[1024];

  unsigned int block_th_id = threadIdx.x;
  std::size_t th_id = blockIdx.x * blockDim.x + block_th_id;

  // Flag = 1 if the current bit is zero.
  unsigned int zero = 0;

  if (th_id < n) {
    // Finds bit is 0 or not
    zero = (((data[th_id] >> bit) & 1u) == 0);
  }

  shared[block_th_id] = zero;
  __syncthreads();

  // Hillis-Steele inclusive prefix sum.
  for (unsigned int offset = 1; offset < blockDim.x; offset *= 2) {

    unsigned int value = shared[block_th_id];

    if (block_th_id >= offset) {
      value += shared[block_th_id - offset];
    }

    __syncthreads();
    shared[block_th_id] = value;
    __syncthreads();
  }

  if (th_id < n) {
    prefix[th_id] = shared[block_th_id];
  }

  // Last thread stores the block's zero count.
  if (block_th_id == blockDim.x - 1) {
    block_totals[blockIdx.x] = shared[block_th_id];
  }
}

/**
 * Swapping numbers in the array.
 */
__global__ void radix_swap(const element_t *input, element_t *output,
                           const unsigned int *prefix,
                           const unsigned int *block_totals, std::size_t n,
                           std::size_t num_blocks, int bit) {
  __shared__ unsigned int block_offset;
  __shared__ unsigned int total_zeros;

  unsigned int block_th_id = threadIdx.x;
  std::size_t th_id = blockIdx.x * blockDim.x + block_th_id;

  // One thread per block sums the block totals.
  if (block_th_id == 0) {
    unsigned int offset = 0;
    unsigned int total = 0;

    for (std::size_t b = 0; b < num_blocks; b++) {
      if (b < blockIdx.x) {
        offset += block_totals[b];
      }

      total += block_totals[b];
    }

    block_offset = offset;
    total_zeros = total;
  }

  __syncthreads();

  if (th_id < n) {
    unsigned int global_prefix = prefix[th_id] + block_offset;

    unsigned int current_bit = (input[th_id] >> bit) & 1u;

    std::size_t destination;

    if (current_bit == 0) {
      destination = global_prefix - 1;
    } else {
      destination = total_zeros + th_id - global_prefix;
    }

    output[destination] = input[th_id];
  }
}

/**
 * Performs all the logic of allocating device vectors and copying host/input
 * vectors to the device. Times the opposing_sort() kernel with wall time,
 * but excludes set up and tear down costs such as mallocs, frees, and memcpies.
 */
void run_gpu_soln(std::vector<element_t> data, std::size_t switch_at,
                  std::size_t n) {

  if (n == 0) {
    return;
  }

  // Used to overwrite the original data back for radix sort
  std::vector<element_t> original = data;

  std::size_t padded_n = 1;

  while (padded_n < n) {
    padded_n *= 2;
  }

  std::vector<element_t> padded(padded_n,
                                std::numeric_limits<element_t>::max());

  std::copy_n(data.begin(), n, padded.begin());

  // Kernel launch configurations. Feel free to change these.
  // This is set to maximise the size of a thread block on a T4, but it hasn't
  // been tuned. It's not known if this is optimal.
  std::size_t const threads_per_block = 1024;

  // Int divison is the floor, this would make it the ceiling
  std::size_t const num_blocks =
      (padded_n + threads_per_block - 1) / threads_per_block;

  // Allocate arrays on the device/GPU
  element_t *d_data;
  cudaMalloc((void **)&d_data, sizeof(element_t) * padded_n);
  CHECK_ERROR("Allocating input array on device");

  // Copy the input from the host to the device/GPU
  cudaMemcpy(d_data, padded.data(), sizeof(element_t) * padded_n,
             cudaMemcpyHostToDevice);
  CHECK_ERROR("Copying input array to device");

  // Warm the cache on the GPU for a more fair comparison
  warm_the_gpu<<<num_blocks, threads_per_block>>>(d_data, switch_at, n);

  cudaDeviceSynchronize();

  // Time the execution of the kernel that you implemented
  auto const kernel_start = std::chrono::high_resolution_clock::now();

  for (std::size_t selection = 2; selection <= padded_n; selection *= 2) {
    for (std::size_t stride = selection / 2; stride > 0; stride /= 2) {
      bitonic_swap<<<num_blocks, threads_per_block>>>(d_data, padded_n, stride,
                                                      selection);
    }
  }

  opposing_sort<<<num_blocks, threads_per_block>>>(d_data, switch_at, n);

  cudaDeviceSynchronize();

  auto const kernel_end = std::chrono::high_resolution_clock::now();

  CHECK_ERROR("Executing kernel on device");

  // After the timer ends, copy the result back, free the device vector,
  // and echo out the timings and the results.
  cudaMemcpy(data.data(), d_data, sizeof(element_t) * n,
             cudaMemcpyDeviceToHost);
  CHECK_ERROR("Transferring result back to host");

  std::cout << "Bitonic time: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                   kernel_end - kernel_start)
                   .count()
            << " ns" << std::endl;

  // for (auto const x : data) {
  //   std::cout << x << " ";
  // }
  // std::cout << std::endl;

  // Radix sort
  std::size_t radix_blocks = (n + threads_per_block - 1) / threads_per_block;

  element_t *d_output;
  unsigned int *d_prefix, *d_block_totals;

  cudaMalloc((void **)&d_output, sizeof(element_t) * n);
  cudaMalloc((void **)&d_prefix, sizeof(unsigned int) * n);
  cudaMalloc((void **)&d_block_totals, sizeof(unsigned int) * radix_blocks);
  CHECK_ERROR("Allocating radix buffers");

  // Reset to the original unsorted input.
  cudaMemcpy(d_data, original.data(), sizeof(element_t) * n,
             cudaMemcpyHostToDevice);
  CHECK_ERROR("Resetting input");

  element_t *input = d_data;
  element_t *output = d_output;

  // Warm the cache on the GPU for a more fair comparison
  warm_the_gpu<<<num_blocks, threads_per_block>>>(d_data, switch_at, n);

  cudaDeviceSynchronize();

  auto const radix_start = std::chrono::high_resolution_clock::now();

  for (int bit = 0; bit < 32; bit++) {
    radix_scan<<<radix_blocks, threads_per_block>>>(input, d_prefix,
                                                    d_block_totals, n, bit);

    radix_swap<<<radix_blocks, threads_per_block>>>(
        input, output, d_prefix, d_block_totals, n, radix_blocks, bit);

    std::swap(input, output);
  }

  opposing_sort<<<radix_blocks, threads_per_block>>>(input, switch_at, n);

  cudaDeviceSynchronize();

  auto const radix_end = std::chrono::high_resolution_clock::now();
  CHECK_ERROR("Executing radix sort");

  std::vector<element_t> radix_result(n);

  cudaMemcpy(radix_result.data(), input, sizeof(element_t) * n,
             cudaMemcpyDeviceToHost);
  CHECK_ERROR("Copying radix result");

  std::cout << "Radix time: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(radix_end -
                                                                    radix_start)
                   .count()
            << " ns\n";

  // for (auto const x : radix_result) {
  //   std::cout << x << " ";
  // }
  // std::cout << std::endl;

  std::cout << "Bitonic vs Radix: " << (data == radix_result ? "PASS" : "FAIL")
            << '\n';

  cudaFree(d_output);
  cudaFree(d_prefix);
  cudaFree(d_block_totals);
  cudaFree(d_data);
  CHECK_ERROR("Freeing GPU memory");
}

} // namespace gpu
} // namespace a1
} // namespace csc485b

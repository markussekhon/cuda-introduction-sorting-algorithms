/**
 * Driver for the benchmark comparison. Generates random data,
 * runs the CPU baseline, and then runs your code.
 */

#include <cstddef>  // std::size_t type
#include <iostream> // std::cout, std::endl
#include <vector>

#include "algorithm_choices.h"
#include "data_generator.h"
#include "data_types.h"

int main() {

  // const std::vector<std::size_t> numbers{4, 5, 7, 8};

  const std::vector<std::size_t> numbers{1024, 2056, 16384, 262144, 1048576};

  for (std::size_t n : numbers) {
    std::size_t const switch_at = 3 * (n >> 2);

    std::cout << "For array size: " << n << std::endl;

    auto data = csc485b::a1::generate_uniform<element_t>(n);
    csc485b::a1::cpu::run_cpu_baseline(data, switch_at, n);
    csc485b::a1::gpu::run_gpu_soln(data, switch_at, n);

    std::cout << "\n" << std::endl;
  }

  return EXIT_SUCCESS;
}

/*! This file is part of davids91/Rafko.
 *
 *    Rafko is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    Rafko is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with Rafko.  If not, see <https://www.gnu.org/licenses/> or
 *    <https://github.com/davids91/rafko/blob/master/LICENSE>
 */

#ifndef RAFKO_GPU_KERNEL_LIBRARY
#define RAFKO_GPU_KERNEL_LIBRARY

#include "rafko_global.hpp"

#include <string>

namespace rafko_utilities {

const std::string atomic_double_add_function = R"(
  #pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

  /* https://suhorukov.blogspot.com/2011/12/opencl-11-atomic-operations-on-floating.html */
  /* https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved */
  inline void AtomicAdd(volatile __global double *source, const double operand) {
    union { unsigned long intVal; double floatVal; } next, expected, current;
    current.floatVal = *source;
    do {
      expected.floatVal = current.floatVal;
      next.floatVal = expected.floatVal + operand;
      current.intVal = atom_cmpxchg((volatile __global unsigned long *)source, expected.intVal, next.intVal);
    } while( current.intVal != expected.intVal );
  }
)";

const std::string random_function = R"(
  /* https://en.wikipedia.org/wiki/Xorshift */
  uint get_random_number(uint range, uint* state){
    uint seed = *state + get_global_id(0);
    uint t = seed ^ (seed << 11);
    uint result = seed ^ (seed >> 19) ^ (t ^ (t >> 8));
    *state = result; /* race condition? */
    return result % range;
  }
)";

const std::string atomic_double_average_function = R"(
  #pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

  inline void AtomicAvg(volatile __global double *source, const double operand) {
    union { unsigned long intVal; double floatVal; } next, expected, current;
    current.floatVal = *source;
    do {
      expected.floatVal = current.floatVal;
      next.floatVal = (expected.floatVal + operand) / 2.0;
      current.intVal = atom_cmpxchg((volatile __global unsigned long *)source, expected.intVal, next.intVal);
    } while( current.intVal != expected.intVal );
  }
)";

} /* namespace rafko_utilities */

#endif /* RAFKO_GPU_KERNEL_LIBRARY */

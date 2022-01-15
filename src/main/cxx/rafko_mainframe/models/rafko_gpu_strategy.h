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

#ifndef RAFKO_GPU_STRATEGY
#define RAFKO_GPU_STRATEGY

#include "rafko_global.h"

#include <utility>
#include <string>
#include <CL/opencl.hpp>

#include "rafko_mainframe/models/rafko_ndarray_shape.h"

namespace rafko_mainframe{

/**
 * @brief      A phase of the Deep learning GPU pipeline consisting of several ordered GPU Kernels.
 */
class RafkoGPUStrategyPhase{
public:
  /**
   * @brief      Provides the kernel source codes of the StrategyPhase in order of execution
   *
   * @return     Vector of {name,source code} pairs in order of intended execution
   */
  virtual std::vector<std::pair<std::string,cl::Program::Sources>> get_steps()const = 0;

  /**
   * @brief      Provides the input dimensions of each step in the Strategy Phase
   *
   * @return     Vector of dimensions in order of @get_steps
   */
  virtual std::vector<RafkoNDArrayShape> get_step_input_dimensions()const = 0;

  /**
   * @brief      Provides the output dimensions of each step in the Strategy Phase
   *
   * @return     Vector of dimensions in order of @get_steps
   */
  virtual std::vector<RafkoNDArrayShape> get_step_output_dimensions()const = 0;

  virtual ~RafkoGPUStrategyPhase() = default;

  bool isValid();
};

} /* namespace rafko_mainframe */

#endif /* RAFKO_GPU_STRATEGY */

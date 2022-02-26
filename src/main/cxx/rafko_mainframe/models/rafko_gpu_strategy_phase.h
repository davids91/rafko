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

#ifndef RAFKO_GPU_STRATEGY_PHASE
#define RAFKO_GPU_STRATEGY_PHASE

#include "rafko_global.h"

#include <utility>
#include <string>
#include <CL/opencl.hpp>

#include "rafko_mainframe/models/rafko_nbuf_shape.h"

namespace rafko_mainframe{

/**
 * @brief      A phase of the Deep learning GPU pipeline strategy describing
 */
class RAFKO_FULL_EXPORT RafkoGPUStrategyPhase{
public:

  /**
   * @brief      Provides the kernel source codes of the StrategyPhase in order of execution
   *
   * @return     Vector of {name,source code} pairs in order of intended execution
   */
  virtual cl::Program::Sources get_step_sources() const = 0;

   /**
    * @brief      Provides the kernel function names of the StrategyPhase
    *
    * @return     Vector of {name,source code} pairs in order of intended execution
    */
  virtual std::vector<std::string> get_step_names() const = 0;

  /**
   * @brief      Provides the input dimensions of each step in the Strategy Phase
   *
   * @return     Vector of dimensions in order of @get_steps
   */
  virtual std::vector<RafkoNBufShape> get_input_shapes() const = 0;

  /**
   * @brief      Provides the output dimensions of each step in the Strategy Phase
   *
   * @return     Vector of dimensions in order of @get_steps
   */
  virtual std::vector<RafkoNBufShape> get_output_shapes() const = 0;

  /**
   * @brief      Provides the required dimensions to solve the phase
   *
   * @return     tuple of {offset, global dimensions, local dimensions}
   */
  virtual std::tuple<cl::NDRange,cl::NDRange,cl::NDRange> get_solution_space() const = 0;

  virtual ~RafkoGPUStrategyPhase() = default;

  bool isValid();
};

} /* namespace rafko_mainframe */

#endif /* RAFKO_GPU_STRATEGY_PHASE */

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

#include "rafko_global.hpp"

#include <utility>
#include <string>
#include <numeric>
#include <CL/opencl.hpp>

#include "rafko_mainframe/models/rafko_nbuf_shape.hpp"

namespace rafko_mainframe{

/**
 * @brief      A phase of the Deep learning GPU pipeline strategy describing the strategy of one entity for
 *             handling GPU operations
 */
class RAFKO_EXPORT RafkoGPUStrategy{
public:
  virtual ~RafkoGPUStrategy() = default;

  /**
   * @brief     provides feedback on whether or not the current returned interfaces
   *            would produce a valid GPU Kernel
   */
  bool isValid() const;


   /**
    * @brief      Provides the kernel function names of the StrategyPhase
    *
    * @return     Vector of {name,source code} pairs in order of intended execution
    */
  virtual std::vector<std::string> get_step_names() const = 0;

  /**
   * @brief      Provides the kernel source codes of the StrategyPhase in order of execution
   *
   * @return     Vector of {name,source code} pairs in order of intended execution
   */
  virtual cl::Program::Sources get_step_sources() const = 0;


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

  /**
   * @brief     Provides the overall size of every component of the input buffer
   *
   * @return    The number of Bytes the whole input buffer occupies
   */
  template<typename T>
  std::uint32_t get_input_buffer_byte_size(){
    std::vector<RafkoNBufShape> input_shapes = get_input_shapes();
    return std::accumulate( input_shapes.begin(), input_shapes.end(), 0.0,
      [](const double& sum, const RafkoNBufShape& object){
        return sum + object.get_byte_size<double>();
      }
    );
  }

  /**
   * @brief     Provides the overall size of every component of the output buffer
   *
   * @return    The number of Bytes the whole output buffer occupies
   */
  template<typename T>
  std::uint32_t get_output_buffer_byte_size(){
    std::vector<RafkoNBufShape> output_shapes = get_output_shapes();
    return std::accumulate( output_shapes.begin(), output_shapes.end(), 0.0,
      [](const double& sum, const RafkoNBufShape& object){
        return sum + object.get_byte_size<T>();
      }
    );
  }
};

} /* namespace rafko_mainframe */

#endif /* RAFKO_GPU_STRATEGY_PHASE */

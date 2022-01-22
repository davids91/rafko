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

#ifndef RAFKO_GPU_PHASE
#define RAFKO_GPU_PHASE

#include "rafko_global.h"

#include <vector>
#include <utility>
#include <string>
#include <CL/opencl.hpp>

#include "rafko_mainframe/models/rafko_nbuf_shape.h"
#include "rafko_mainframe/models/rafko_gpu_strategy_phase.h"

namespace rafko_mainframe{

/**
 * @brief      A phase of the Deep learning GPU pipeline consisting of several ordered GPU Kernels.
 */
class RafkoGPUPhase{
public:
  RafkoGPUPhase(
    cl::Context& context_, cl::Device& device_, cl::CommandQueue& queue_,
    std::shared_ptr<RafkoGPUStrategyPhase> strategy_
  ):opencl_context(context_)
  , opencl_device(device_)
  , opencl_device_queue(queue_)
  { set_strategy(strategy_); }

  /**
   * @brief      Implements a GPU Strategy phase provided in the argument
   *
   * @param      strategy_ the strategy parts to implenet in this phase
   */
  void set_strategy(std::shared_ptr<RafkoGPUStrategyPhase> strategy_);

  /**
   * @brief      Executes the implemented strategy phase
   *
   * @param[in]  enq      the OpenCL enqeue arguments provided by the caller
   * @param[in]  input    the input array to upload to device memory
   */
  void operator()(cl::EnqueueArgs& enq, const std::vector<sdouble32>& input);

  /**
   * @brief      Executes the implemented strategy phase
   *
   * @param[in]  enq      the OpenCL enqeue arguments provided by the caller
   * @param[in]  input    the input buffer containing input data already on device
   */
  void operator()(cl::EnqueueArgs& enq, cl::Buffer& input);

  /**
   * @brief      Constructs a buffer containing the output data of the implemented strategy phase
   *
   * @param[in]  size     The number of elements to take from the output
   * @param[in]  offset   An offset inside the output buffer to get
   *
   * @return     The output data with ownership transferred to the caller
   */
  std::unique_ptr<sdouble32[]> acquire_output(std::size_t size, std::size_t offset = 0u);

  /**
   * @brief      Loads the output of the Phase into the supported pointer
   *
   * @param[in]  target     The pointer to load the output data into
   * @param[in]  size     The number of elements to take from the output
   * @param[in]  offset   An offset inside the output buffer to get
   */
  void load_output(sdouble32* target, std::size_t size, std::size_t offset = 0u);

  /**
   * @brief      Provides the buffer containing the output data of the implemented
   *             strategy phase inside device memory
   *
   * @return     A buffer of the output data on the device
   */
  cl::Buffer& get_output_buffer(){
    return std::get<0>(kernel_args.back());
  }

private:
  cl::Context& opencl_context;
  cl::Device& opencl_device;
  cl::CommandQueue& opencl_device_queue;
  std::shared_ptr<RafkoGPUStrategyPhase> strategy;
  std::vector<std::tuple<cl::Buffer, cl::Buffer, int>> kernel_args;
  std::vector<cl::KernelFunctor<cl::Buffer, cl::Buffer, int, cl::Buffer, cl::Buffer, int>> steps;
};

} /* namespace rafko_mainframe */

#endif /* RAFKO_GPU_PHASE */

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

#include "rafko_global.hpp"

#include <CL/opencl.hpp>
#include <string>
#include <utility>
#include <vector>

#include "rafko_mainframe/models/rafko_gpu_strategy.hpp"
#include "rafko_mainframe/models/rafko_nbuf_shape.hpp"

namespace rafko_mainframe {

/**
 * @brief      A phase of the Deep learning GPU pipeline consisting of several
 * ordered GPU Kernels.
 */
class RAFKO_EXPORT RafkoGPUPhase {
public:
  RafkoGPUPhase(const cl::Context &context, const cl::Device &device,
                cl::CommandQueue &queue,
                std::shared_ptr<RafkoGPUStrategy> strategy)
      : m_openclContext(context), m_openclDevice(device),
        m_openclDeviceQueue(queue) {
    set_strategy(strategy);
  }

  /**
   * @brief      Implements a GPU Strategy phase provided in the argument
   *
   * @param      strategy the strategy parts to implenet in this phase
   */
  void set_strategy(std::shared_ptr<RafkoGPUStrategy> strategy);

  /**
   * @brief      Executes the implemented strategy phase
   *
   * @param[in]  input    the input array to upload to device memory
   */
  void operator()(const std::vector<double> &input);

  /**
   * @brief      Executes the implemented strategy phase
   *
   * @param[in]  input    the input buffer containing input data already on
   * device
   */
  void operator()(cl::Buffer &input);

  /**
   * @brief      Executes the implemented strategy phase
   *
   */
  void operator()();

  /**
   * @brief      Executes the implemented strategy phase
   *
   * @param[in]  enq      the OpenCL enqeue arguments provided by the caller
   * @param[in]  input    the input array to upload to device memory
   */
  void operator()(cl::EnqueueArgs enq, const std::vector<double> &input);

  /**
   * @brief      Executes the implemented strategy phase
   *
   * @param[in]  enq      the OpenCL enqeue arguments provided by the caller
   * @param[in]  input    the input buffer containing input data already on
   * device
   */
  void operator()(cl::EnqueueArgs enq, cl::Buffer &input);

  /**
   * @brief      Executes the implemented strategy phase
   *
   * @param[in]  enq      the OpenCL enqeue arguments provided by the caller
   */
  void operator()(cl::EnqueueArgs enq);

  /**
   * @brief      Constructs a buffer containing the output data of the
   * implemented strategy phase
   *
   * @param[in]  size     The number of elements to take from the output
   * @param[in]  offset   An offset inside the output buffer to get
   *
   * @return     The output data with ownership transferred to the caller
   */
  std::unique_ptr<double[]> acquire_output(std::size_t size,
                                           std::size_t offset = 0u) const;

  /**
   * @brief      Loads the output of the Phase into the supported pointer
   *
   * @param[in]  target   The pointer to load the output data into
   * @param[in]  size     The number of elements to take from the output
   * @param[in]  offset   An offset inside the output buffer to get
   */
  void load_output(double *target, std::size_t size,
                   std::size_t offset = 0u) const;

  /**
   * @brief      Provides the buffer containing the input data of the
   * implemented strategy phase inside device memory to upload data to
   *
   * @return     A buffer of the input data on the device
   */
  cl::Buffer &get_input_buffer() { return std::get<0>(m_kernelArgs.front()); }

  /**
   * @brief      Provides the buffer containing the output data of the
   * implemented strategy phase inside device memory
   *
   * @return     A buffer of the output data on the device
   */
  cl::Buffer &get_output_buffer() { return std::get<0>(m_kernelArgs.back()); }

private:
  using KernelFunctor = cl::KernelFunctor<cl::Buffer, cl::Buffer, int,
                                          cl::Buffer, cl::Buffer, int>;
  const cl::Context &m_openclContext;
  const cl::Device &m_openclDevice;
  cl::CommandQueue &m_openclDeviceQueue;
  std::shared_ptr<RafkoGPUStrategy> m_strategy;
  std::vector<std::tuple<cl::Buffer, cl::Buffer, int>> m_kernelArgs;
  std::vector<KernelFunctor> m_steps;
};

} /* namespace rafko_mainframe */

#endif /* RAFKO_GPU_PHASE */

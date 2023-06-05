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

#include "rafko_mainframe/services/rafko_gpu_phase.hpp"

#include <iostream>
#include <stdexcept>

#include "rafko_mainframe/services/rafko_assertion_logger.hpp"
#include "rafko_utilities/services/rafko_string_utils.hpp"

namespace rafko_mainframe {

void RafkoGPUPhase::set_strategy(std::shared_ptr<RafkoGPUStrategy> strategy) {
  RFASSERT_SCOPE(STRATEGY_BUILD);
  RFASSERT_LOG("Setting GPU Strategy phase..");
  RFASSERT(strategy->isValid());
  m_strategy = strategy;
  m_kernelArgs.clear();
  m_steps.clear();

  cl_int return_value;
  std::vector<std::string> names = m_strategy->get_step_names();
  cl::Program::Sources sources = m_strategy->get_step_sources();
  std::vector<RafkoNBufShape> input_shapes = m_strategy->get_input_shapes();
  std::vector<RafkoNBufShape> output_shapes = m_strategy->get_output_shapes();
  std::vector<cl::Event> dimension_write_events(names.size() + 1u);
  /*!Note: Since steps in the phase share buffers, where one steps input is the
   * preceeding steps output, overall to write the dimensions @names.size() + 1
   * events need to be generated to fill up buffer dimension data, because
   * there's an input for each step plus there's an output for the last step.
   * The vector shall not be updated beyond this point to make sure that
   * internal re-alignment will not happen during the operations, during which a
   * pointer is being taken from each element
   */

  /* Compile Kernel program */
  cl::Program program(m_openclContext, sources);
  return_value = program.build({m_openclDevice}, "-cl-std=CL2.0");
  std::string build_log =
      program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_openclDevice);
  if (return_value != CL_SUCCESS) {
    RFASSERT_LOG("OpenCL Kernel Compilation failed with log: {}", build_log);
    throw std::runtime_error("OpenCL Kernel compilation failed with error: \n" +
                             build_log + "\n");
  }
  if (0 < build_log.length()) {
    RFASSERT_LOG("OpenCL kernel compilation successful! Log: {}", build_log);
  } else {
    RFASSERT_LOG("OpenCL kernel compilation successful!");
  }

  /* Set buffers, kernel arguments and functors */
  std::uint32_t step_index = 0;
  for (const std::string &step_name : names) {
    RFASSERT_LOG("Assembling step {} of strategy phase: input bytesize: {}; "
                 "input dimensions byte size: {}",
                 step_index, input_shapes[step_index].get_byte_size<double>(),
                 input_shapes[step_index].get_shape_buffer_byte_size());
    m_kernelArgs.push_back(
        std::make_tuple(/* push in each step input */
                        cl::Buffer(
                            m_openclContext, CL_MEM_READ_WRITE,
                            input_shapes[step_index].get_byte_size<double>()),
                        cl::Buffer(m_openclContext, CL_MEM_READ_ONLY,
                                   input_shapes[step_index]
                                       .get_shape_buffer_byte_size()),
                        input_shapes[step_index].size()));
    std::unique_ptr<cl_int[]> shape_data =
        input_shapes[step_index].acquire_shape_buffer();
    RFASSERT_LOGV(input_shapes[step_index], "Uploading buffer dimensions: ");
    return_value =
        m_openclDeviceQueue
            .enqueueWriteBuffer(/* upload buffer dimensions */
                                std::get<1>(m_kernelArgs.back()), CL_FALSE,
                                0u /*offset*/,
                                input_shapes[step_index]
                                    .get_shape_buffer_byte_size(),
                                shape_data.get(), NULL /*events(to wait for)*/,
                                &dimension_write_events[step_index]);
    if (CL_SUCCESS != return_value) {
      RFASSERT_LOG("OpenCL Return value: {}", return_value);
    }
    RFASSERT(return_value == CL_SUCCESS);
    m_steps.push_back(KernelFunctor(program, step_name));
    ++step_index;
  }
  RFASSERT_LOGV(output_shapes.back(),
                "Assembling output buffer of strategy phase: bytesize: {}; "
                "Output Shape: ",
                output_shapes.back().get_byte_size<double>());
  m_kernelArgs.push_back(std::make_tuple(
      cl::Buffer(m_openclContext, CL_MEM_READ_WRITE,
                 output_shapes.back().get_byte_size<double>()),
      cl::Buffer(m_openclContext, CL_MEM_READ_ONLY,
                 output_shapes.back().get_shape_buffer_byte_size()),
      output_shapes.back().size()));
  std::unique_ptr<cl_int[]> shape_data =
      output_shapes.back().acquire_shape_buffer();
  return_value =
      m_openclDeviceQueue
          .enqueueWriteBuffer(/* upload buffer dimensions */
                              std::get<1>(m_kernelArgs.back()), CL_FALSE,
                              0u /*offset*/,
                              output_shapes.back().get_shape_buffer_byte_size(),
                              shape_data.get(), NULL /*events(to wait for)*/,
                              &dimension_write_events[names.size()]);
  if (CL_SUCCESS != return_value) {
    RFASSERT_LOG("OpenCL Return value: {}", return_value);
  }
  RFASSERT(return_value == CL_SUCCESS);

  for (cl::Event &event : dimension_write_events) {
    return_value = event.wait();
    if (CL_SUCCESS != return_value) {
      RFASSERT_LOG("OpenCL Return value: {}", return_value);
    }
    RFASSERT(return_value == CL_SUCCESS);
  }
}

void RafkoGPUPhase::operator()(const std::vector<double> &input) {
  (*this)(std::make_from_tuple<cl::EnqueueArgs>(std::tuple_cat(
              std::tie(m_openclDeviceQueue), m_strategy->get_solution_space())),
          input);
}

void RafkoGPUPhase::operator()(cl::Buffer &input) {
  (*this)(std::make_from_tuple<cl::EnqueueArgs>(std::tuple_cat(
              std::tie(m_openclDeviceQueue), m_strategy->get_solution_space())),
          input);
}

void RafkoGPUPhase::operator()() {
  (*this)(std::make_from_tuple<cl::EnqueueArgs>(std::tuple_cat(
      std::tie(m_openclDeviceQueue), m_strategy->get_solution_space())));
}

void RafkoGPUPhase::operator()(cl::EnqueueArgs enq,
                               const std::vector<double> &input) {
  RafkoNBufShape input_shape = m_strategy->get_input_shapes()[0];
  RFASSERT_LOG("Number of inputs: {} vs. {} (byte size: {})",
               input_shape.get_number_of_elements(), input.size(),
               input_shape.get_byte_size<double>());
  RFASSERT(input_shape.get_number_of_elements() == input.size());

  cl::Buffer input_buf_cl(m_openclContext, CL_MEM_READ_ONLY,
                          input_shape.get_byte_size<double>());
  cl_int return_value =
      m_openclDeviceQueue
          .enqueueWriteBuffer(/* upload input to device memory */
                              input_buf_cl, CL_TRUE /*blocking*/, 0 /*offset*/,
                              input_shape.get_byte_size<double>() /*size*/,
                              input.data());
  if (CL_SUCCESS != return_value) {
    RFASSERT_LOG("OpenCL Return value: {}", return_value);
  }
  RFASSERT(return_value == CL_SUCCESS);
  (*this)(enq, input_buf_cl);
}

void RafkoGPUPhase::operator()(cl::EnqueueArgs enq, cl::Buffer &input) {
  [[maybe_unused]] cl_int return_value =
      m_steps[0](enq, input, std::get<1>(m_kernelArgs[0]),
                 std::get<2>(m_kernelArgs[0]), std::get<0>(m_kernelArgs[1]),
                 std::get<1>(m_kernelArgs[1]), std::get<2>(m_kernelArgs[1]))
          .wait();
  if (CL_SUCCESS != return_value) {
    RFASSERT_LOG("OpenCL Return value: {}", return_value);
  }
  RFASSERT(return_value == CL_SUCCESS);
  for (std::uint32_t step_index = 1; step_index < m_steps.size();
       ++step_index) {
    return_value =
        m_steps[step_index](enq, std::get<0>(m_kernelArgs[step_index - 1]),
                            std::get<1>(m_kernelArgs[step_index - 1]),
                            std::get<2>(m_kernelArgs[step_index - 1]),
                            std::get<0>(m_kernelArgs[step_index]),
                            std::get<1>(m_kernelArgs[step_index]),
                            std::get<2>(m_kernelArgs[step_index]))
            .wait();
    if (CL_SUCCESS != return_value) {
      RFASSERT_LOG("OpenCL Return value: {}", return_value);
    }
    RFASSERT(return_value == CL_SUCCESS);
    ++step_index;
  }
}

void RafkoGPUPhase::operator()(cl::EnqueueArgs enq) {
  [[maybe_unused]] cl_int return_value =
      m_steps[0](enq, std::get<0>(m_kernelArgs[0]),
                 std::get<1>(m_kernelArgs[0]), std::get<2>(m_kernelArgs[0]),
                 std::get<0>(m_kernelArgs[1]), std::get<1>(m_kernelArgs[1]),
                 std::get<2>(m_kernelArgs[1]))
          .wait();
  if (CL_SUCCESS != return_value) {
    RFASSERT_LOG("OpenCL Return value: {}", return_value);
  }
  RFASSERT(return_value == CL_SUCCESS);
  for (std::uint32_t step_index = 1; step_index < m_steps.size();
       ++step_index) {
    return_value =
        m_steps[step_index](enq, std::get<0>(m_kernelArgs[step_index - 1]),
                            std::get<1>(m_kernelArgs[step_index - 1]),
                            std::get<2>(m_kernelArgs[step_index - 1]),
                            std::get<0>(m_kernelArgs[step_index]),
                            std::get<1>(m_kernelArgs[step_index]),
                            std::get<2>(m_kernelArgs[step_index]))
            .wait();
    if (CL_SUCCESS != return_value) {
      RFASSERT_LOG("OpenCL Return value: {}", return_value);
    }
    RFASSERT(return_value == CL_SUCCESS);
    ++step_index;
  }
}

std::unique_ptr<double[]>
RafkoGPUPhase::acquire_output(std::size_t size, std::size_t offset) const {
  RafkoNBufShape output_shape = m_strategy->get_output_shapes().back();

  RFASSERT_LOG("Acquiring output[{} + {}]", offset, size,
               output_shape.get_number_of_elements());
  RFASSERT((offset + size) <= output_shape.get_number_of_elements());

  std::unique_ptr<double[]> output(new double[size]);
  load_output(output.get(), size, offset);
  return output;
}

void RafkoGPUPhase::load_output(double *target, std::size_t size,
                                std::size_t offset) const {
  RafkoNBufShape output_shape = m_strategy->get_output_shapes().back();
  RFASSERT_LOG("Loading output[{} + {}]", offset, size,
               output_shape.get_number_of_elements());
  RFASSERT(nullptr != target);
  RFASSERT((sizeof(double) * size) <= output_shape.get_byte_size<double>());

  const cl::Buffer &output_buffer_cl = std::get<0>(m_kernelArgs.back());
  [[maybe_unused]] cl_int return_value = m_openclDeviceQueue.enqueueReadBuffer(
      output_buffer_cl, CL_TRUE /*blocking*/, (sizeof(double) * offset),
      (sizeof(double) * size), target);
  if (CL_SUCCESS != return_value) {
    RFASSERT_LOG("OpenCL Return value: {}", return_value);
  }
  RFASSERT(return_value == CL_SUCCESS);
}

} // namespace rafko_mainframe

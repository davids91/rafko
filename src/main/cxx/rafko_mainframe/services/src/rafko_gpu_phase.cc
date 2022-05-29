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

#include "rafko_mainframe/services/rafko_gpu_phase.h"

#include <stdexcept>
#include <iostream>

#include "rafko_utilities/services/rafko_string_utils.h"
#include "rafko_mainframe/services/rafko_assertion_logger.h"

namespace rafko_mainframe{

void RafkoGPUPhase::set_strategy(std::shared_ptr<RafkoGPUStrategyPhase> strategy_){
  RFASSERT_LOG("Setting GPU Strategy phase..");
  RFASSERT( strategy_->isValid() );
  strategy = strategy_;
  kernel_args.clear();
  steps.clear();

  cl_int return_value;
  std::vector<std::string> names = strategy->get_step_names();
  cl::Program::Sources sources = strategy->get_step_sources();
  std::vector<RafkoNBufShape> input_shapes = strategy->get_input_shapes();
  std::vector<RafkoNBufShape> output_shapes = strategy->get_output_shapes();
  std::vector<cl::Event> dimension_write_events(names.size() + 1u);
  /*!Note: Since steps in the phase share buffers, where one steps input is the
   * preceeding steps output, overall to write the dimensions @names.size() + 1 events
   * need to be generated to fill up buffer dimension data, because there's an input for
   * each step plus there's an output for the last step.
   * The vector shall not be updated beyond this point to make sure that internal re-alignment
   * will not happen during the operations, during which a pointer is being taken from each element
   */

  /* Compile Kernel program */
  cl::Program program(opencl_context, sources);
  return_value = program.build({opencl_device});
  if(return_value != CL_SUCCESS){
    std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(opencl_device);
    RFASSERT_LOG("{}", build_log);
    throw std::runtime_error( "OpenCL Kernel compilation failed with error: \n" + build_log + "\n" );
  }

  /* Set buffers, kernel arguments and functors */
  std::uint32_t step_index = 0;
  for(const std::string& step_name : names){
    RFASSERT_LOG(
      "Assembling step {} of strategy phase: input bytesize: {}; input dimensions byte size: {}",
      step_index, input_shapes[step_index].get_byte_size<double>(), input_shapes[step_index].get_shape_buffer_byte_size()
    );
    kernel_args.push_back(std::make_tuple( /* push in each step input */
      cl::Buffer(opencl_context, CL_MEM_READ_WRITE, input_shapes[step_index].get_byte_size<double>()),
      cl::Buffer(opencl_context, CL_MEM_READ_ONLY, input_shapes[step_index].get_shape_buffer_byte_size()),
      input_shapes[step_index].size()
    ));
    RFASSERT_LOGV(input_shapes[step_index], "Uploading buffer dimensions: ");
    return_value = opencl_device_queue.enqueueWriteBuffer( /* upload buffer dimensions */
      std::get<1>(kernel_args.back()), CL_FALSE, 0u/*offset*/,
      input_shapes[step_index].get_shape_buffer_byte_size(),
      input_shapes[step_index].acquire_shape_buffer().get(),
      NULL/*events(to wait for)*/, &(*(dimension_write_events.begin() + step_index))
    );
    RFASSERT( return_value == CL_SUCCESS );
    steps.push_back(cl::KernelFunctor<cl::Buffer, cl::Buffer, int, cl::Buffer, cl::Buffer, int>(
      program, step_name
    ));
    ++step_index;
  }
  RFASSERT_LOG(
    "Assembling output buffer of strategy phase: bytesize: {}",
    output_shapes.back().get_byte_size<double>()
  );
  kernel_args.push_back(std::make_tuple(
    cl::Buffer(opencl_context, CL_MEM_READ_WRITE, output_shapes.back().get_byte_size<double>()),
    cl::Buffer(opencl_context, CL_MEM_READ_ONLY, output_shapes.back().get_shape_buffer_byte_size()),
    output_shapes.back().size()
  ));
  return_value = opencl_device_queue.enqueueWriteBuffer( /* upload buffer dimensions */
    std::get<1>(kernel_args.back()), CL_FALSE, 0u/*offset*/,
    output_shapes.back().get_shape_buffer_byte_size(),
    output_shapes.back().acquire_shape_buffer().get(),
    NULL/*events(to wait for)*/, &(*(dimension_write_events.begin() + names.size()))
  );
  RFASSERT( return_value == CL_SUCCESS );

  for(cl::Event& event : dimension_write_events){
    return_value = event.wait();
    RFASSERT( return_value == CL_SUCCESS );
  }

}

void RafkoGPUPhase::operator()(cl::EnqueueArgs& enq, const std::vector<double>& input){
  RafkoNBufShape input_shape = strategy->get_input_shapes()[0];
  RFASSERT_LOG(
    "Number of inputs: {} vs. {}",
    input_shape.get_number_of_elements(), input.size()
  );
  RFASSERT( input_shape.get_number_of_elements() == input.size() );

  cl::Buffer input_buf_cl( opencl_context, CL_MEM_READ_ONLY, input_shape.get_byte_size<double>() );
  cl_int return_value = opencl_device_queue.enqueueWriteBuffer( /* upload input to device memory */
    input_buf_cl, CL_TRUE/*blocking*/,
    0/*offset*/, input_shape.get_byte_size<double>()/*size*/,
    input.data()
  );
  RFASSERT( return_value == CL_SUCCESS );

  (*this)(enq, input_buf_cl);
}

void RafkoGPUPhase::operator()(cl::EnqueueArgs& enq, cl::Buffer& input){
  steps[0](enq,
    input, std::get<1>(kernel_args[0]), std::get<2>(kernel_args[0]),
    std::get<0>(kernel_args[1]), std::get<1>(kernel_args[1]), std::get<2>(kernel_args[1])
  ).wait();
  for(std::uint32_t step_index = 1; step_index < steps.size(); ++step_index){
    steps[step_index](enq,
      std::get<0>(kernel_args[step_index - 1]), std::get<1>(kernel_args[step_index - 1]), std::get<2>(kernel_args[step_index - 1]),
      std::get<0>(kernel_args[step_index]), std::get<1>(kernel_args[step_index]), std::get<2>(kernel_args[step_index])
    ).wait();
    ++step_index;
  }
}

void RafkoGPUPhase::operator()(cl::EnqueueArgs& enq){
  steps[0](enq,
    std::get<0>(kernel_args[0]), std::get<1>(kernel_args[0]), std::get<2>(kernel_args[0]),
    std::get<0>(kernel_args[1]), std::get<1>(kernel_args[1]), std::get<2>(kernel_args[1])
  ).wait();
  for(std::uint32_t step_index = 1; step_index < steps.size(); ++step_index){
    steps[step_index](enq,
      std::get<0>(kernel_args[step_index - 1]), std::get<1>(kernel_args[step_index - 1]), std::get<2>(kernel_args[step_index - 1]),
      std::get<0>(kernel_args[step_index]), std::get<1>(kernel_args[step_index]), std::get<2>(kernel_args[step_index])
    ).wait();
    ++step_index;
  }
}

std::unique_ptr<double[]> RafkoGPUPhase::acquire_output(std::size_t size, std::size_t offset){
  RafkoNBufShape output_shape = strategy->get_output_shapes().back();

  RFASSERT_LOG("Acquiring output[{} + {}] / ", offset, size, output_shape.get_number_of_elements());
  RFASSERT( (offset + size) <= output_shape.get_number_of_elements() );

  std::unique_ptr<double[]> output(new double[size]);
  load_output(output.get(), size, offset);
  return output;
}

void RafkoGPUPhase::load_output(double* target, std::size_t size, std::size_t offset){
  RafkoNBufShape output_shape = strategy->get_output_shapes().back();

  RFASSERT_LOG("Loading output[{} + {}] / ", offset, size, output_shape.get_number_of_elements());
  RFASSERT( (sizeof(double) * size) <= output_shape.get_byte_size<double>() );

  cl::Buffer& output_buffer_cl = std::get<0>(kernel_args.back());
  cl_int return_value = opencl_device_queue.enqueueReadBuffer(
    output_buffer_cl, CL_TRUE/*blocking*/,
    (sizeof(double) * offset), (sizeof(double) * size),
    target
  );
  RFASSERT( return_value == CL_SUCCESS );
}


} /* rafko_mainframe */

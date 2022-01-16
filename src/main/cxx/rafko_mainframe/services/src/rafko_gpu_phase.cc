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

#include <assert.h>

namespace rafko_mainframe{

void RafkoGPUPhase::set_strategy(std::shared_ptr<RafkoGPUStrategyPhase> strategy_){
  assert( strategy_->isValid() );
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
  assert( return_value == CL_SUCCESS );

  /* Set buffers, kernel arguments and functors */
  uint32 step_index = 0;
  for(const std::string& step_name : names){
    kernel_args.push_back(std::make_tuple( /* push in each step input */
      cl::Buffer(opencl_context, CL_MEM_READ_WRITE, input_shapes[step_index].get_byte_size<sdouble32>()),
      cl::Buffer(opencl_context, CL_MEM_READ_ONLY, input_shapes[step_index].get_shape_buffer_byte_size()),
      input_shapes[step_index].size()
    ));
    return_value = opencl_device_queue.enqueueWriteBuffer( /* upload buffer dimensions */
      std::get<1>(kernel_args.back()), CL_FALSE, 0,
      input_shapes[step_index].get_shape_buffer_byte_size(),
      input_shapes[step_index].acquire_shape_buffer().get(),
      NULL/*events(to wait for)*/, &(*(dimension_write_events.begin() + step_index))
    );
    assert( return_value == CL_SUCCESS );
    steps.push_back(cl::KernelFunctor<cl::Buffer, cl::Buffer, int, cl::Buffer, cl::Buffer, int>(
      program, step_name
    ));
    ++step_index;
  }
  kernel_args.push_back(std::make_tuple(
    cl::Buffer(opencl_context, CL_MEM_READ_WRITE, output_shapes.back().get_byte_size<sdouble32>()),
    cl::Buffer(opencl_context, CL_MEM_READ_ONLY, output_shapes.back().get_shape_buffer_byte_size()),
    output_shapes.back().size()
  ));
  return_value = opencl_device_queue.enqueueWriteBuffer( /* upload buffer dimensions */
    std::get<1>(kernel_args.back()), CL_FALSE, 0,
    output_shapes.back().get_shape_buffer_byte_size(),
    output_shapes.back().acquire_shape_buffer().get(),
    NULL/*events(to wait for)*/, &(*(dimension_write_events.begin() + names.size()))
  );
  assert( return_value == CL_SUCCESS );

  for(cl::Event& event : dimension_write_events){
    return_value = event.wait();
    assert( return_value == CL_SUCCESS );
  }

}

void RafkoGPUPhase::operator()(cl::EnqueueArgs& enq, const std::vector<sdouble32>& input){
  RafkoNBufShape input_shape = strategy->get_input_shapes()[0];

  assert( input_shape.get_number_of_elements() == input.size() );

  cl::Buffer input_buf_cl( opencl_context, CL_MEM_READ_ONLY, input_shape.get_byte_size<sdouble32>() );
  cl_int return_value = opencl_device_queue.enqueueWriteBuffer( /* upload input to device memory */
    input_buf_cl, CL_TRUE, 0, input_shape.get_byte_size<sdouble32>(), input.data()
  );
  assert( return_value == CL_SUCCESS );

  (*this)(enq, input_buf_cl);
}

void RafkoGPUPhase::operator()(cl::EnqueueArgs& enq, cl::Buffer& input){
  steps[0](enq,
    input, std::get<1>(kernel_args[0]), std::get<2>(kernel_args[0]),
    std::get<0>(kernel_args[1]), std::get<1>(kernel_args[1]), std::get<2>(kernel_args[1])
  );
  for(uint32 step_index = 1; step_index < steps.size(); ++step_index){
    steps[step_index](enq,
      std::get<0>(kernel_args[step_index - 1]), std::get<1>(kernel_args[step_index - 1]), std::get<2>(kernel_args[step_index - 1]),
      std::get<0>(kernel_args[step_index]), std::get<1>(kernel_args[step_index]), std::get<2>(kernel_args[step_index])
    ).wait();
    ++step_index;
  }
}

std::unique_ptr<sdouble32[]> RafkoGPUPhase::acquire_output(){
  RafkoNBufShape output_shape = strategy->get_output_shapes().back();
  assert( (sizeof(sdouble32) * output_shape.get_number_of_elements()) == output_shape.get_byte_size<sdouble32>() );

  std::unique_ptr<sdouble32[]> output(new sdouble32[output_shape.get_number_of_elements()]);
  load_output(output.get(), output_shape.get_number_of_elements());
  return output;
}

void RafkoGPUPhase::load_output(sdouble32* target, std::size_t size){
  RafkoNBufShape output_shape = strategy->get_output_shapes().back();
  assert( (sizeof(sdouble32) * size) == output_shape.get_byte_size<sdouble32>() );

  cl::Buffer& output_buffer_cl = std::get<0>(kernel_args.back());
  cl_int return_value = opencl_device_queue.enqueueReadBuffer( /* download last output from device memory */
    output_buffer_cl, CL_TRUE, 0, output_shape.get_byte_size<sdouble32>(), target
  );
  assert( return_value == CL_SUCCESS );
}


} /* rafko_mainframe */

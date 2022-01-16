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

#ifndef RAFKO_GPU_PIPELINE
#define RAFKO_GPU_PIPELINE

#include "rafko_global.h"

#include <utility>
#include <string>
#include <CL/opencl.hpp>

#include "rafko_mainframe/models/rafko_nbuf_shape.h"
#include "rafko_mainframe/models/rafko_gpu_strategy_phase.h"

namespace rafko_mainframe{

/**
 * @brief      Deep learning pipeline to handle buffers through feature solve to error calculation
 */
class RafkoGPUPipeline{
public:
  RafkoGPUPipeline(){
    
  }
private:
  cl::Context& opencl_context;
  cl::Device& opencl_device;
  cl::CommandQueue& opencl_device_queue;

  std::pair<RafkoNBufShape,cl::Buffer> weights_and_inputs;
  std::pair<RafkoNBufShape,cl::Buffer> features_and_labels;
  std::pair<RafkoNBufShape,cl::Buffer> error_value;
  RafkoGPUPhase solution_phase;
  RafkoGPUPhase error_phase;

};

} /* namespace rafko_mainframe */

#endif /* RAFKO_GPU_PIPELINE */

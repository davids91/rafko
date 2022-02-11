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

#include "rafko_mainframe/models/rafko_gpu_strategy_phase.h"

#include  <iostream>

namespace rafko_mainframe{

bool RafkoGPUStrategyPhase::isValid(){
  std::vector<std::string> step_names = get_step_names();
  std::vector<RafkoNBufShape> input_shapes = get_input_shapes();
  std::vector<RafkoNBufShape> output_shapes = get_output_shapes();

  if(
    (0u == step_names.size())
    ||(step_names.size() != get_step_sources().size())
    ||(step_names.size() != input_shapes.size())
    ||(step_names.size() != output_shapes.size())
  ) return false;

  for(sint32 dimension_index = 0; dimension_index < (static_cast<sint32>(input_shapes.size()) - 1); ++dimension_index){
    if(input_shapes[dimension_index + 1] != output_shapes[dimension_index])
      return false;
  }

  return true;
}

} /* rafko_mainframe */

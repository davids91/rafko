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

 #include "rafko_mainframe/models/rafko_gpu_strategy.h"

namespace rafko_mainframe{

bool RafkoGPUStrategyPhase::isValid(){
  if(get_step_input_dimensions().size() != get_step_output_dimensions().size())
    return false;

  if(0 == get_step_input_dimensions().size())
    return false;

  for(uint32 dimension_index = 0; dimension_index < (get_step_input_dimensions().size() - 1); ++dimension_index){
    if(get_step_output_dimensions()[dimension_index] != get_step_input_dimensions()[dimension_index + 1])
      return false;
  }
  return true;
}

} /* rafko_mainframe */

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

#include "sparse_net_library/services/weight_experience_space.h"

#include <stdexcept>
#include <cmath>

#include <iostream>

namespace sparse_net_library{

using std::abs;

Weight_experience_space::Weight_experience_space(sdouble32 weight_min_, sdouble32 weight_max_, sdouble32 weight_step_)
:  weight_min(weight_min_)
,  weight_max(weight_max_)
,  weight_step(weight_step_)
,  weight_values((weight_max - weight_min)/weight_step)
,  experiences((weight_max - weight_min)/weight_step)
,  best_weight_index(0)
,  smallest_experience(0)
{
  if(weight_min > weight_max)
    throw std::runtime_error("Minimum value can not be greater, than maximum value!");

  if(0 == weight_values.size())
    throw std::runtime_error("Unable to build space with the given resolution!");

  sdouble32 weight_value = weight_min;
  for(uint32 weight_index = 0; weight_index < weight_values.size(); ++weight_index){
    weight_values[weight_index] = weight_value;
    weight_value += weight_step;
  }
}

sdouble32 Weight_experience_space::add_experience(sdouble32 value){
  experiences[best_weight_index] += value;
  if(abs(experiences[best_weight_index]) < abs(experiences[smallest_experience]))
    smallest_experience = best_weight_index;

  find_best_weight();
  cut();
  return weight_values[best_weight_index];
}

void Weight_experience_space::find_best_weight(void){
  best_weight_index = 0;
  for(uint32 weight_index = 1; weight_index < experiences.size(); ++weight_index){
    if(experiences[weight_index] > experiences[best_weight_index])
      best_weight_index = weight_index;
  }
}

void Weight_experience_space::cut(void){
  for(uint32 weight_index = 1; weight_index < experiences.size(); ++weight_index){
    std::cout << "abs(" << experiences[weight_index] <<") - abs(" << experiences[weight_index] << ")" <<std::endl;
    experiences[weight_index] = std::copysign(
      (abs(experiences[weight_index]) - abs(experiences[smallest_experience])),
      experiences[weight_index]
    );
  }
}

sdouble32 Weight_experience_space::get_best_weight(void){
  return weight_values[best_weight_index];
}


} /* namespace sparse_net_library */
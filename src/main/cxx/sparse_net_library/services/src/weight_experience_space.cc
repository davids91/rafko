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

#include <cmath>
#include <stdexcept>
#include <iostream>
namespace sparse_net_library{

using std::abs;
using std::max;

Weight_experience_space::Weight_experience_space(sdouble32 weight_min_, sdouble32 weight_max_, sdouble32 weight_step_)
:  weight_min(weight_min_)
,  weight_max(weight_max_)
,  weight_step(weight_step_)
,  weight_values(1 + (weight_max - weight_min)/weight_step)
,  experiences(1 + (weight_max - weight_min)/weight_step)
,  best_weight_index(rand()%(weight_values.size()))
,  worst_weight_index(rand()%(weight_values.size()))
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
  if(abs(experiences[best_weight_index]) < abs(experiences[smallest_experience])){
    smallest_experience = best_weight_index;
    cut();
  }
  evaluate_weights();
  adapt_weight(worst_weight_index);
  return weight_values[best_weight_index];
}

void Weight_experience_space::adapt_weight(uint32 weight_index){
  if( /* Only adapt the weights not on the edge of the space, to preserve the range of the space */
    (0 < weight_index)&&(weight_index < (weight_values.size()-1))
    &&(experiences[weight_index] < experiences[weight_index - 1]) /* And only adapt the weight if both the left */
    &&(experiences[weight_index] < experiences[weight_index + 1]) /* and right weights have better experiences, than it does */
  ){
    /* Offset the neighbours experience vectors with the smallest of the 3 weight experiences, so every experience value would be in positive range */
    sdouble32 left_weight_xp = experiences[weight_index-1] + experiences[weight_index];
    sdouble32 right_weight_xp = experiences[weight_index+1] + experiences[weight_index];

    /* Then normalize the weight experience values */
    sdouble32 max_xp = max(abs(left_weight_xp),abs(right_weight_xp));
    left_weight_xp /= max_xp;
    right_weight_xp /= max_xp;

    /* update the worst weight value based on a weighthed average */
    weight_values[weight_index] = (
      (weight_values[weight_index-1] * left_weight_xp)
      + (weight_values[weight_index+1] * right_weight_xp)
    ) / (left_weight_xp + right_weight_xp);

    if(
      (weight_values[weight_index] < weight_values[0])
      ||(weight_values[weight_index] > weight_values[weight_values.size()-1])
    ){
      std::cout << "WHAT" << std::endl;
      std::cout << weight_values[weight_index]
      << " = ((" << weight_values[weight_index-1] << "*((" << experiences[weight_index-1] << "+" << experiences[weight_index] << ")/" << max_xp << "))"
      << " + (" << weight_values[weight_index+1] << "*((" << experiences[weight_index+1] << "+" << experiences[weight_index] << ")/" << max_xp <<")))"
      << "/(" << left_weight_xp << "+" << right_weight_xp << ")"
      << std::endl;
      std::cout << "===" << std::endl;
    }

    /*!Note: Since the first and last weight values are never touched, the weight experience space remains intact.
     *       The weights inside the space might shimmy, to look for better performing weight values.
     */
  }
}

void Weight_experience_space::evaluate_weights(void){
  best_weight_index = 0;
  worst_weight_index = 0;
  for(uint32 weight_index = 1; weight_index < experiences.size(); ++weight_index){
    if(experiences[weight_index] > experiences[best_weight_index])
      best_weight_index = weight_index;
    if(experiences[weight_index] < experiences[worst_weight_index]){
      worst_weight_index = weight_index;
    }
  }
}

void Weight_experience_space::cut(void){
  for(uint32 weight_index = 1; weight_index < experiences.size(); ++weight_index){
    experiences[weight_index] = std::copysign(
      (abs(experiences[weight_index]) - abs(experiences[smallest_experience])), experiences[weight_index]
    );
  }
}

sdouble32 Weight_experience_space::get_best_weight(void){
  return weight_values[best_weight_index];
}


} /* namespace sparse_net_library */
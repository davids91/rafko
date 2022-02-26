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
#include "rafko_gym/services/weight_updater_amsgrad.h"

namespace rafko_gym{

void RafkoWeightUpdaterAMSGrad::iterate(const std::vector<sdouble32>& gradients){
  sdouble32 raw_moment;
  for(uint32 weight_index = 0; weight_index < moment.size(); ++weight_index){
    moment[weight_index] = (
      (settings.get_beta() * moment[weight_index])
      + ( ((double_literal(1.0) - settings.get_beta()) * gradients[weight_index]) )
    );
    raw_moment = (
      (settings.get_beta_2() * raw_moment_max[weight_index])
      + (
        ((double_literal(1.0) - settings.get_beta_2())
        * std::pow(gradients[weight_index], double_literal(2.0)))
      )
    );
    if(raw_moment > raw_moment_max[weight_index])
      raw_moment_max[weight_index] = raw_moment;
  }
  RafkoWeightUpdater::iterate(gradients);
  ++iteration_count;
}

} /* namespace rafko_gym */

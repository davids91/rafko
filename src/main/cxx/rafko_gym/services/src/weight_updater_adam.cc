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
#include "rafko_gym/services/weight_updater_adam.h"

namespace rafko_gym{

void RafkoWeightUpdaterAdam::iterate(const std::vector<double>& gradients){
  for(std::uint32_t weight_index = 0; weight_index < moment.size(); ++weight_index){
    moment[weight_index] = (
      (settings.get_beta() * moment[weight_index])
      + ( (((1.0) - settings.get_beta()) * gradients[weight_index]) )
    );
    raw_moment[weight_index] = (
      (settings.get_beta_2() * raw_moment[weight_index])
      + (
        (((1.0) - settings.get_beta_2())
        * std::pow(gradients[weight_index], (2.0)))
      )
    );
  }
  RafkoWeightUpdater::iterate(gradients);
  ++iteration_count;
}

double RafkoWeightUpdaterAdam::get_new_velocity(std::uint32_t weight_index, const std::vector<double>& gradients){
  parameter_not_used(gradients); /* the variable moment contains the processed value of the gradients, so no need to use it here again. */
  return (
    settings.get_learning_rate() / (
      std::sqrt(
        raw_moment[weight_index] / (
          (1.0) - std::pow(
             settings.get_beta(), static_cast<double>(iteration_count)
         )
        )
      ) + settings.get_epsilon()
    ) * (
      moment[weight_index] / (
        (1.0) - std::pow(
          settings.get_beta_2(), static_cast<double>(iteration_count)
        )
      )
    )
  );
}

} /* namespace rafko_gym */

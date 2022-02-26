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

#ifndef WEIGHT_UPDATER_MOMENTUM_H
#define WEIGHT_UPDATER_MOMENTUM_H

#include "rafko_gym/services/rafko_weight_updater.h"

#include <vector>

namespace rafko_gym{

class RAFKO_FULL_EXPORT RafkoWeightUpdaterMomentum : public RafkoWeightUpdater{
public:
  RafkoWeightUpdaterMomentum(rafko_net::RafkoNet& rafko_net, rafko_net::Solution& solution_, const rafko_mainframe::RafkoSettings& settings_)
  :  RafkoWeightUpdater(rafko_net, solution_, settings_)
  ,  previous_velocity(rafko_net.weight_table_size(),(0.0))
  { }

  void iterate(const std::vector<double>& gradients){
    RafkoWeightUpdater::iterate(gradients);
    std::copy(get_current_velocity().begin(),get_current_velocity().end(),previous_velocity.begin());
  }

private:
  double get_new_velocity(std::uint32_t weight_index, const std::vector<double>& gradients){
    return (
      (previous_velocity[weight_index] * settings.get_gamma())
      + (gradients[weight_index] * settings.get_learning_rate())
    );
  }

  std::vector<double> previous_velocity;
};

} /* namespace rafko_gym */

#endif /* WEIGHT_UPDATER_MOMENTUM_H */

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

#ifndef WEIGHT_UPDATER_NESTEROVS_H
#define WEIGHT_UPDATER_NESTEROVS_H

#include "rafko_gym/services/rafko_weight_updater.h"

namespace rafko_gym{

class RAFKO_FULL_EXPORT RafkoWeightUpdaterNesterovs : public RafkoWeightUpdater{
public:
  RafkoWeightUpdaterNesterovs(rafko_net::RafkoNet& rafko_net, rafko_net::Solution& solution_, rafko_mainframe::RafkoSettings& settings_)
  :  RafkoWeightUpdater(rafko_net, solution_, settings_, 2)
  { }

  void iterate(const std::vector<sdouble32>& gradients){
    RafkoWeightUpdater::iterate(gradients);
    std::copy(get_current_velocity().begin(),get_current_velocity().end(),previous_velocity.begin());
  }

  void start(){
    RafkoWeightUpdater::start();
    std::copy(get_current_velocity().begin(),get_current_velocity().end(),previous_velocity_at_start.begin());
  }

private:
  sdouble32 get_new_velocity(uint32 weight_index, const std::vector<sdouble32>& gradients){
    if(!is_finished()) return (
      (previous_velocity[weight_index] * settings.get_gamma())
      + (gradients[weight_index] * settings.get_learning_rate())
    );
    else return(
      (previous_velocity_at_start[weight_index] * settings.get_gamma())
      + (gradients[weight_index] * settings.get_learning_rate())
    );
  }

  std::vector<sdouble32> previous_velocity_at_start;
  std::vector<sdouble32> previous_velocity;
};

} /* namespace rafko_gym */

#endif /* WEIGHT_UPDATER_NESTEROVS_H */

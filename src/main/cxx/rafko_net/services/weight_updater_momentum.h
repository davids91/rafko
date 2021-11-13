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

#include "rafko_net/services/weight_updater.h"

namespace rafko_net{

class WeightUpdaterMomentum : public WeightUpdater{
public:
  WeightUpdaterMomentum(RafkoNet& rafko_net, Solution& solution_, ServiceContext& service_context_)
  :  WeightUpdater(rafko_net, solution_, service_context_)
  ,  previous_velocity(rafko_net.weight_table_size(),double_literal(0.0))
  { }

  void iterate(const vector<sdouble32>& gradients){
    WeightUpdater::iterate(gradients);
    std::copy(get_current_velocity().begin(),get_current_velocity().end(),previous_velocity.begin());
  }

private:
  sdouble32 get_new_velocity(uint32 weight_index, const vector<sdouble32>& gradients){
    return (
      (previous_velocity[weight_index] * service_context.get_gamma())
      + (gradients[weight_index] * service_context.get_learning_rate())
    );
  }

  vector<sdouble32> previous_velocity;
};

} /* namespace rafko_net */

#endif /* WEIGHT_UPDATER_MOMENTUM_H */

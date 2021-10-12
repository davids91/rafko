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

#ifndef WEIGHT_UPDATER_AMSGRAD_H
#define WEIGHT_UPDATER_AMSGRAD_H

#include "rafko_net/services/weight_updater.h"

namespace sparse_net_library{

class Weight_updater_amsgrad : public Weight_updater{
public:
  Weight_updater_amsgrad(SparseNet& sparse_net, Service_context& service_context_)
  :  Weight_updater(sparse_net, service_context_)
  ,  iteration_count(0)
  ,  moment(sparse_net.weight_table_size(),double_literal(0.0))
  ,  raw_moment_max(sparse_net.weight_table_size(),double_literal(0.0))
  { }

  void iterate(const vector<sdouble32>& gradients,Solution& solution){
    sdouble32 raw_moment;
    for(uint32 weight_index = 0; weight_index < moment.size(); ++weight_index){
      moment[weight_index] = (
        (service_context.get_beta() * moment[weight_index])
        + ( ((double_literal(1.0) - service_context.get_beta()) * gradients[weight_index]) )
      );
      raw_moment = (
        (service_context.get_beta_2() * raw_moment_max[weight_index])
        + (
          ((double_literal(1.0) - service_context.get_beta_2())
          * std::pow(gradients[weight_index], double_literal(2.0)))
        )
      );
      if(raw_moment > raw_moment_max[weight_index])
        raw_moment_max[weight_index] = raw_moment;
    }
    Weight_updater::iterate(gradients, solution);
    ++iteration_count;
  }

private:
  sdouble32 get_new_velocity(uint32 weight_index, const vector<sdouble32>& gradients){
    return ( service_context.get_step_size() * moment[weight_index] / ( std::sqrt(raw_moment_max[weight_index]) + service_context.get_epsilon() ) );
  }

  uint32 iteration_count;

  vector<sdouble32> moment;
  vector<sdouble32> raw_moment_max;
};

} /* namespace sparse_net_library */

#endif /* WEIGHT_UPDATER_AMSGRAD_H */

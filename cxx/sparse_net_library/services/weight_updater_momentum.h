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

#include "sparse_net_library/services/weight_updater.h"

namespace sparse_net_library{

class Weight_updater_momentum : public Weight_updater{
public: 
  Weight_updater_momentum(SparseNet& sparse_net, Service_context service_context = Service_context())
  :  Weight_updater(sparse_net, service_context)
  ,  previous_velocity(sparse_net.weight_table_size(),double_literal(0.0))
  { }

  void iterate(vector<unique_ptr<atomic<sdouble32>>>& gradients,Solution& solution){
    Weight_updater::iterate(gradients, solution);
    std::copy(get_current_velocity().begin(),get_current_velocity().end(),previous_velocity.begin());
  }

protected: 
  sdouble32 get_new_velocity(uint32 weight_index, const vector<unique_ptr<atomic<sdouble32>>>& gradients){
    return (
      (previous_velocity[weight_index] * context.get_gamma()) 
      + (*gradients[weight_index] * context.get_step_size())
    );
  }
private:
  vector<sdouble32> previous_velocity;
};

} /* namespace sparse_net_library */

#endif /* WEIGHT_UPDATER_MOMENTUM_H */
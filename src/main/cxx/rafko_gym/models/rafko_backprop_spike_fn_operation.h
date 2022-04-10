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

#ifndef RAFKO_BACKPROP_SPIKE_FN_OPERATION_H
#define RAFKO_BACKPROP_SPIKE_FN_OPERATION_H

#include "rafko_global.h"

#include <vector>
#include <memory>
#include <utility>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/services/rafko_assertion_logger.h"
#include "rafko_net/models/spike_function.h"

#include "rafko_gym/models/rafko_backpropagation_operation.h"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_FULL_EXPORT RafkoBackpropSpikeFnOperation{
public:
  RafkoBackpropNetworkInputOperation(
    RafkoBackPropagation& queue, const rafko_net::RafkoNet& network,
    std::uint32_t past_index, std::uint32_t neuron_index_
  ):RafkoBackpropagationOperation(queue, network, past_index)
  , neuron_index(neuron_index_)
  {
  }

  void upload_dependencies_to_operations(){
    present_value_dependency = push_dependency(ad_operation_neuron_transfer_d, past_index, neuron_index);
    if(past_index < network.memory_size()){
      past_value_dependency = push_dependency(ad_operation_neuron_spike_d, (past_index + 1u), neuron_index);
    }
    set_registered()
  }

  void calculate(
    std::uint32 d_w_index, std::uint32 run_index,
    const std::vector<std::vector<double>>& network_input, const std::vector<std::vector<double>>& label_data
  ){
    RFASSERT(run_index < network_input.size());
    RFASSERT(run_index < label_data.size());
    if(past_index > run_index){
      value = 0.0;
      derivative_value = 0.0;
    }else{
      double past_value = (past_value_dependency)?(past_value_dependency->get_value()):(0.0);
      value = SpikeFunction::get_value(
        network.neuron_array(neuron_index).spike_function(),
        network.weight_table(network.neuron_array(neuron_index).input_weights(0).starts()),
        present_value_dependency->get_value(), past_value
      );
      double past_derivative_value = (past_value_dependency)?(past_value_dependency->get_derivative()):(0.0);
      if(d_w_index == network.neuron_array(neuron_index).input_weights(0).starts()){
        derivative_value = SpikeFunction::get_derivative_for_w(
          network.neuron_array(neuron_index).spike_function(),
          present_value_dependency->get_value(), present_value_dependency->get_derivative(),
          past_value, past_derivative_value
        );
      }else{
        derivative_value = SpikeFunction::get_derivative_not_for_w(
          network.neuron_array(neuron_index).spike_function(),
          present_value_dependency->get_value(), present_value_dependency->get_derivative(),
          past_value, past_derivative_value
        );
      }
    }
    set_processed();
  }
private:
  const std::uint32_t neuron_index;
  std::shared_ptr<RafkoBackpropagationOperation> present_value_dependency;
  std::shared_ptr<RafkoBackpropagationOperation> past_value_dependency;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_SPIKE_FN_OPERATION_H */

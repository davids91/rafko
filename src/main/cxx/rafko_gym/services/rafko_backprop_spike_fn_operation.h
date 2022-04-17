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

#include "rafko_gym/services/rafko_backpropagation_operation.h"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_FULL_EXPORT RafkoBackpropSpikeFnOperation
: public RafkoBackpropagationOperation
{
public:
  RafkoBackpropSpikeFnOperation(
    RafkoBackPropagationData& data, const rafko_net::RafkoNet& network,
    std::uint32_t operation_index, std::uint32_t neuron_index_
  ):RafkoBackpropagationOperation(data, network, operation_index)
  , neuron_index(neuron_index_)
  {
  }

  DependencyRequest upload_dependencies_to_operations(){
    return {{
      {{ad_operation_neuron_transfer_d, {neuron_index}}},
      [this](std::vector<std::shared_ptr<RafkoBackpropagationOperation>> dependencies){
        RFASSERT(1 == dependencies.size());
        present_value_dependency = dependencies[0];
        set_registered();
      }
    }};
  }

  void calculate(
    std::uint32_t d_w_index, std::uint32_t run_index,
    const std::vector<std::vector<double>>& network_input,
    const std::vector<std::vector<double>>& label_data
  ){
    RFASSERT(run_index < network_input.size());
    RFASSERT(run_index < label_data.size());
    RFASSERT(static_cast<bool>(present_value_dependency));
    double past_value = (1u <= run_index)?(get_value(run_index - 1u)):(0.0);
    double past_derivative_value = (1u <= run_index)?(get_derivative((run_index - 1u), d_w_index)):(0.0);
    set_value(run_index, rafko_net::SpikeFunction::get_value(
      network.neuron_array(neuron_index).spike_function(),
      network.weight_table(network.neuron_array(neuron_index).input_weights(0).starts()),
      present_value_dependency->get_value(run_index), past_value
    ));
    if(static_cast<std::int32_t>(d_w_index) == network.neuron_array(neuron_index).input_weights(0).starts()){
      set_derivative(run_index, d_w_index, rafko_net::SpikeFunction::get_derivative_for_w(
        network.neuron_array(neuron_index).spike_function(),
        network.weight_table(network.neuron_array(neuron_index).input_weights(0).starts()),
        present_value_dependency->get_value(run_index),
        present_value_dependency->get_derivative(run_index, d_w_index),
        past_value, past_derivative_value
      ));
    }else{
      set_derivative(run_index, d_w_index, rafko_net::SpikeFunction::get_derivative_not_for_w(
        network.neuron_array(neuron_index).spike_function(),
        network.weight_table(network.neuron_array(neuron_index).input_weights(0).starts()),
        present_value_dependency->get_value(run_index),
        present_value_dependency->get_derivative(run_index, d_w_index),
        past_value, past_derivative_value
      ));
    }
    set_processed();
  }
private:
  const std::uint32_t neuron_index;
  std::shared_ptr<RafkoBackpropagationOperation> present_value_dependency;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_SPIKE_FN_OPERATION_H */

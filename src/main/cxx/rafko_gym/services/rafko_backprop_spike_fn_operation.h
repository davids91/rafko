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
  )
  : RafkoBackpropagationOperation(data, network, operation_index)
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

  void calculate_value(const std::vector<double>& network_input, const std::vector<double>& label_data){
    parameter_not_used(network_input);
    parameter_not_used(label_data);
    RFASSERT(are_dependencies_registered());
    RFASSERT(static_cast<bool>(present_value_dependency));
    RFASSERT(present_value_dependency->is_value_processed());
    double past_value = get_value(1u/*past_index*/);
    set_value(rafko_net::SpikeFunction::get_value(
      network.neuron_array(neuron_index).spike_function(),
      network.weight_table(network.neuron_array(neuron_index).input_weights(0).starts()),
      present_value_dependency->get_value(0u/*past_index*/), past_value
    ));
    set_value_processed();
  }

  void calculate_derivative(
    std::uint32_t d_w_index, const std::vector<double>& network_input, const std::vector<double>& label_data
  ){
    parameter_not_used(network_input);
    parameter_not_used(label_data);
    RFASSERT(is_value_processed());
    RFASSERT(are_dependencies_registered());
    RFASSERT(static_cast<bool>(present_value_dependency));
    RFASSERT(present_value_dependency->is_processed());
    double past_value = get_value(1u/*past_index*/);
    double past_derivative_value = get_derivative(1u/*past_index*/, d_w_index);
    if(static_cast<std::int32_t>(d_w_index) == network.neuron_array(neuron_index).input_weights(0).starts()){
      set_derivative(d_w_index, rafko_net::SpikeFunction::get_derivative_for_w(
        network.neuron_array(neuron_index).spike_function(),
        network.weight_table(network.neuron_array(neuron_index).input_weights(0).starts()),
        present_value_dependency->get_value(0u/*past_index*/),
        present_value_dependency->get_derivative(0u/*past_index*/, d_w_index),
        past_value, past_derivative_value
      ));
    }else{
      set_derivative(d_w_index, rafko_net::SpikeFunction::get_derivative_not_for_w(
        network.neuron_array(neuron_index).spike_function(),
        network.weight_table(network.neuron_array(neuron_index).input_weights(0).starts()),
        present_value_dependency->get_value(0u/*past_index*/),
        present_value_dependency->get_derivative(0u/*past_index*/, d_w_index),
        past_value, past_derivative_value
      ));
    }
    set_derivative_processed();
  }

  #if(RAFKO_USES_OPENCL)
  std::string value_kernel_function() const{
    RFASSERT(static_cast<bool>(present_value_dependency));
    RFASSERT(present_value_dependency->are_dependencies_registered());
    return (
      "Spike[" + std::to_string(neuron_index) + "]:"
      + rafko_net::Spike_functions_Name(network.neuron_array(neuron_index).spike_function()) + "\n"
      + present_value_dependency-> value_kernel_function()
    );
  }
  std::string derivative_kernel_function() const{
    return "";
  }
  #endif/*(RAFKO_USES_OPENCL)*/

  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> get_dependencies(){
    return {present_value_dependency};
  }

private:
  const std::uint32_t neuron_index;
  std::shared_ptr<RafkoBackpropagationOperation> present_value_dependency;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_SPIKE_FN_OPERATION_H */

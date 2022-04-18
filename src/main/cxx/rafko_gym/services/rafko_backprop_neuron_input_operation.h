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

#ifndef RAFKO_BACKPROP_NEURON_INPUT_OPERATION_H
#define RAFKO_BACKPROP_NEURON_INPUT_OPERATION_H

#include "rafko_global.h"

#include <vector>
#include <memory>
#include <utility>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/services/rafko_assertion_logger.h"
#include "rafko_net/models/input_function.h"
#include "rafko_net/services/synapse_iterator.h"

#include "rafko_gym/services/rafko_backpropagation_operation.h"
#include "rafko_gym/services/rafko_backprop_network_input_operation.h"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_FULL_EXPORT RafkoBackpropNeuronInputOperation
: public RafkoBackpropagationOperation
{
public:
  RafkoBackpropNeuronInputOperation(
    RafkoBackPropagationData& data, const rafko_net::RafkoNet& network,
    std::uint32_t operation_index, std::uint32_t neuron_index_, std::uint32_t neuron_input_index_
  )
  : RafkoBackpropagationOperation(data, network, operation_index)
  , neuron_index(neuron_index_)
  , neuron_input_index(neuron_input_index_)
  , inputs_iterator(network.neuron_array(neuron_index).input_indices())
  , weights_iterator(network.neuron_array(neuron_index).input_weights())
  , is_network_input(
    rafko_net::SynapseIterator<rafko_net::InputSynapseInterval>::is_index_input(
      inputs_iterator[neuron_input_index]
    )
  )
  , input_index_from_neuron_input_index(
    (!is_network_input)?(inputs_iterator[neuron_input_index])
    :(
      rafko_net::SynapseIterator<rafko_net::InputSynapseInterval>::synapse_index_from_input_index(
        inputs_iterator[neuron_input_index]
      )
    )
  )
  , input_past_index(
    inputs_iterator.reach_past_loops<rafko_net::InputSynapseInterval>(neuron_input_index)
  )
  , weight_index(weights_iterator[1u + neuron_input_index]) /* spike index preceeds the inputs */
  {
  }

  DependencyRequest upload_dependencies_to_operations(){
    DependencyParameters dependency_parameters;
    if(is_network_input){ /* weighted pair from a Neuron or a Network input */
      RFASSERT(0u == input_past_index);
      dependency_parameters.push_back({
        ad_operation_network_input_d,
        {
          input_index_from_neuron_input_index, /*!Note: Network input dependency contains weight */
          static_cast<std::uint32_t>(weights_iterator[1 + neuron_input_index])
        }
      });
    }else{ /* if it's not an input, then it's an internal neuron value */
      dependency_parameters.push_back({ad_operation_neuron_spike_d, {input_index_from_neuron_input_index}});
    }
    if(neuron_input_index < (inputs_iterator.cached_size() - 1u)){ /* this is not the last input */
      /* push in dependency u(x) = every input after this one */
      dependency_parameters.push_back({
        ad_operation_neuron_input_d, {neuron_index, (neuron_input_index + 1u)}
      });
      /*!Note: current operation is to calculate the inputs starting from the current index,
       * but the elements starting from the next input is as a dependency.
       */
    }else{ /* this is the last input, push in the bias dependency */
      dependency_parameters.push_back({
        ad_operation_neuron_bias_d, {neuron_index, (1u + neuron_input_index + 1u)}
      });
    }

    return {{dependency_parameters, [this](std::vector<std::shared_ptr<RafkoBackpropagationOperation>> dependencies){
        RFASSERT(1 <= dependencies.size());
        if(is_network_input)
          network_input_dependency = dependencies[0];
          else neuron_data_dependency = dependencies[0];
        if(neuron_input_index < (inputs_iterator.cached_size() - 1u)){
          RFASSERT(2 == dependencies.size());
          neuron_input_dependency = dependencies[1];
        }else{
          RFASSERT(2 == dependencies.size());
          neuron_bias_dependency = dependencies[1];
        }
        set_registered();
      }
    }};
  }

  void calculate(
    std::uint32_t d_w_index, std::uint32_t run_index,
    const std::vector<std::vector<double>>& network_input, const std::vector<std::vector<double>>& label_data
  ){
    /* i(w) = w * f(w) ¤ u(w) | f(w) = network_input or input_from_internal_neuron */
    RFASSERT(run_index < network_input.size());
    RFASSERT(run_index < label_data.size());
    RFASSERT(are_dependencies_registered());
    /* calculate f(x) part */
    double weighted_input;
    double current_input_derivative;
    if(is_network_input){
      RFASSERT(0u == input_past_index);
      RFASSERT(static_cast<bool>(network_input_dependency));
      RFASSERT(network_input_dependency->is_processed());
      weighted_input = network_input_dependency->get_value(run_index);
      current_input_derivative = network_input_dependency->get_derivative(run_index, d_w_index);
    }else{
      RFASSERT(static_cast<bool>(neuron_data_dependency));
      RFASSERT(neuron_data_dependency->is_processed());
      if(input_index_from_neuron_input_index <= run_index){
        weighted_input = (
          neuron_data_dependency->get_value(run_index - input_index_from_neuron_input_index)
          * network.weight_table(weight_index)
        );
        //TODO: re-check derivative formula here
        current_input_derivative = (
          neuron_data_dependency->get_derivative(run_index - input_index_from_neuron_input_index, d_w_index)
          * network.weight_table(weight_index)
        );
        if(weight_index == d_w_index)current_input_derivative += weighted_input;
      }else{
        weighted_input = 0.0;
        current_input_derivative = 0.0;
      }
    }

    /* calculate u(x) part, u(x) is either the inputs starting from the next, or the bias value(s) */
    double next_value = 0.0;
    double next_derivative = 0.0;
    // std::cout << "neuron[" << neuron_index << "], input[" << neuron_input_index << "]:";
    if(neuron_input_index < (inputs_iterator.cached_size() - 1u)){
      RFASSERT(static_cast<bool>(neuron_input_dependency));
      RFASSERT(neuron_input_dependency->is_processed());
      next_value = neuron_input_dependency->get_value(run_index);
      next_derivative = neuron_input_dependency->get_derivative(run_index, d_w_index);
    }else{ /* the last input starts to collect bias */
      RFASSERT(static_cast<bool>(neuron_bias_dependency));
      RFASSERT(neuron_bias_dependency->is_processed());
      next_value = neuron_bias_dependency->get_value(run_index);
      next_derivative = neuron_bias_dependency->get_derivative(run_index, d_w_index);
    }

    /* calculate the overall value and derivative part */
    // std::cout << weighted_input << " * " << next_value << std::endl;
    set_value(run_index, rafko_net::InputFunction::collect(
      network.neuron_array(neuron_index).input_function(),
      weighted_input, next_value
    ));
    set_derivative(run_index, d_w_index, rafko_net::InputFunction::get_derivative(
      network.neuron_array(neuron_index).input_function(),
      weighted_input, current_input_derivative, next_value, next_derivative
    ));
    set_processed();
  }

  #if(RAFKO_USES_OPENCL)
  std::string value_kernel_function() const{
    std::string value_debug;
    if(0u == neuron_input_index){ /* the first input!*/
      value_debug = (
        "|| \t input_function[" + std::to_string(neuron_index) + "]: "
        + rafko_net::Input_functions_Name(network.neuron_array(neuron_index).input_function()) + "\n"
      );
    }
    if(is_network_input){
      RFASSERT(static_cast<bool>(network_input_dependency));
      RFASSERT(network_input_dependency->are_dependencies_registered());
      value_debug += (
        "|| \t --> " + network_input_dependency->value_kernel_function() + "\n"
      );
    }else{
      RFASSERT(static_cast<bool>(neuron_data_dependency));
      RFASSERT(neuron_data_dependency->are_dependencies_registered());
      value_debug += (
        "|| \t --> " + neuron_data_dependency->value_kernel_function() + "\n"
      );
    }
    if(neuron_input_index < (inputs_iterator.cached_size() - 1u)){ /* not the last input */
      RFASSERT(static_cast<bool>(neuron_input_dependency));
      RFASSERT(neuron_input_dependency->are_dependencies_registered());
      value_debug += (
        "|| \t ---> \n" + neuron_input_dependency->value_kernel_function()
      );
    }else{
      RFASSERT(static_cast<bool>(neuron_bias_dependency));
      RFASSERT(neuron_bias_dependency->are_dependencies_registered());
      value_debug += (
        "|| \t ---> \n" + neuron_bias_dependency->value_kernel_function()
      );
    }
    return value_debug;
  }
  std::string derivative_kernel_function() const{
    return "";
  }
  #endif/*(RAFKO_USES_OPENCL)*/

  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> get_dependencies(){
    return {network_input_dependency, neuron_data_dependency, neuron_input_dependency};
  }

private:
  const std::uint32_t neuron_index;
  const std::uint32_t neuron_input_index;
  rafko_net::SynapseIterator<rafko_net::InputSynapseInterval> inputs_iterator;
  rafko_net::SynapseIterator<rafko_net::IndexSynapseInterval> weights_iterator;

  const bool is_network_input;
  const std::uint32_t input_index_from_neuron_input_index;
  const std::uint32_t input_past_index;
  const std::uint32_t weight_index;

  std::shared_ptr<RafkoBackpropagationOperation> network_input_dependency;
  std::shared_ptr<RafkoBackpropagationOperation> neuron_data_dependency;
  std::shared_ptr<RafkoBackpropagationOperation> neuron_input_dependency;
  std::shared_ptr<RafkoBackpropagationOperation> neuron_bias_dependency;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_NEURON_INPUT_OPERATION_H */

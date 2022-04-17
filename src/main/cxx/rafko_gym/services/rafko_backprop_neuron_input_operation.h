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
  ):RafkoBackpropagationOperation(data, network, operation_index)
  , neuron_index(neuron_index_)
  , neuron_input_index(neuron_input_index_)
  , inputs_iterator(network.neuron_array(neuron_index).input_indices())
  , weights_iterator(network.neuron_array(neuron_index).input_weights())
  , is_network_input(
    rafko_net::SynapseIterator<rafko_net::InputSynapseInterval>::is_index_input(
      inputs_iterator[neuron_input_index]
    )
  ),input_index_from_neuron_input_index(
    (!is_network_input)?(inputs_iterator[neuron_input_index])
    :(
      rafko_net::SynapseIterator<rafko_net::InputSynapseInterval>::synapse_index_from_input_index(
        inputs_iterator[neuron_input_index]
      )
    )
  ),input_past_index(
    inputs_iterator.reach_past_loops<rafko_net::InputSynapseInterval>(neuron_input_index)
  ),weight_index(weights_iterator[1u + neuron_input_index]) /* spike index preceeds the inputs */
  {
  }

  DependencyRequest upload_dependencies_to_operations(){
    DependencyParameters dependency_parameters;
    if(is_network_input){ /* weighted pair from a Neuron or a Network input */
      RFASSERT(0u == input_past_index);
      dependency_parameters.push_back({
        ad_operation_network_input_d,
        {
          input_index_from_neuron_input_index,
          static_cast<std::uint32_t>(weights_iterator[1 + neuron_input_index])
        }
      });
    }else{ /* if it's not an input, then it's an internal neuron value */
      dependency_parameters.push_back({ad_operation_neuron_spike_d,{input_index_from_neuron_input_index}});
    }
    if(neuron_input_index < (inputs_iterator.cached_size() - 1u)){
      /* push in dependency u(x) = every input after this one */
      dependency_parameters.push_back({
        ad_operation_neuron_input_d,{neuron_index, (neuron_input_index + 1)}
      });
      /*!Note: current operation is to calculate the inputs starting from the current index,
       * but the elements starting from the next input is as a dependency.
       */
    }

    return {{dependency_parameters, [this](std::vector<std::shared_ptr<RafkoBackpropagationOperation>> dependencies){
        if(is_network_input)
          network_input_dependency = dependencies[0];
          else neuron_data_dependency = dependencies[0];
        if(neuron_input_index < (inputs_iterator.cached_size() - 1u)){
          RFASSERT(2 == dependencies.size());
          neuron_input_dependency = dependencies[1];
        }
        set_registered();
      }
    }};
  }

  void calculate(
    std::uint32_t d_w_index, std::uint32_t run_index,
    const std::vector<std::vector<double>>& network_input, const std::vector<std::vector<double>>& label_data
  ){
    RFASSERT(run_index < network_input.size());
    RFASSERT(run_index < label_data.size());
    RFASSERT(are_dependencies_registered());
    if(neuron_input_index < (inputs_iterator.cached_size() - 1u)){ /* not the last input */
      /* i(w) = w * f(w) ¤ u(w) | f(w) = network_input or input_from_internal_neuron */
      RFASSERT(static_cast<bool>(neuron_input_dependency));
      RFASSERT(neuron_input_dependency->is_processed());
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
      set_value(run_index, rafko_net::InputFunction::collect(
        network.neuron_array(neuron_index).input_function(),
        weighted_input, neuron_input_dependency->get_value(run_index)
      ));
      set_derivative(run_index, d_w_index, rafko_net::InputFunction::get_derivative(
        network.neuron_array(neuron_index).input_function(),
        weighted_input, current_input_derivative,
        neuron_input_dependency->get_value(run_index),
        neuron_input_dependency->get_derivative(run_index, d_w_index)
      ));
    }else{ /* the last input: i(w) = w * f(w) */
      double current_input_derivative;
      double weighted_input;
      if(is_network_input){ /*!Note: Network input dependency contains weight */
        RFASSERT(static_cast<bool>(network_input_dependency));
        RFASSERT(network_input_dependency->is_processed());
        weighted_input = network_input[run_index][input_index_from_neuron_input_index];
        current_input_derivative = network_input_dependency->get_derivative(run_index, d_w_index);
      }else{
        RFASSERT(static_cast<bool>(neuron_data_dependency));
        RFASSERT(neuron_data_dependency->is_processed());
        if(input_index_from_neuron_input_index <= run_index){
          weighted_input = (
            neuron_data_dependency->get_value(run_index - input_index_from_neuron_input_index)
            * network.weight_table(weight_index)
          );
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
      set_value(run_index, weighted_input);
      set_derivative(run_index, d_w_index, current_input_derivative);
    }
    set_processed();
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
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_NEURON_INPUT_OPERATION_H */

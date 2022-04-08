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

#ifndef RAFKO_BACKPROP_NETWORK_INPUT_OPERATION_H
#define RAFKO_BACKPROP_NETWORK_INPUT_OPERATION_H

#include "rafko_global.h"

#include <vector>
#include <memory>
#include <utility>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/services/rafko_assertion_logger.h"
#include "rafko_net/models/spike_function.h"
#include "rafko_net/services/synapse_iterator.h"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_FULL_EXPORT RafkoBackpropNetworkInputOperation{
public:
  RafkoBackpropNetworkInputOperation(
    rafko_net::RafkoNet& network, std::uint32_t backprop_index
    std::vector<std::unique_ptr<RafkoBackpropagationOperation>>& operations,
  ): RafkoBackpropagationOperation(network, backprop_index, operations)
  {
  }

  void upload_dependencies_to_operations(){
    switch(operation){
      case ad_operation_objective_d:{
        std::uint32_t neuron_index = (network.neuron_array_size() - network.output_neuron_number() + operation_index);
        dependencies.push_back(operations.size());
        operations.emplace_back(
          network, operations, neuron_index, ad_operation_neuron_spike_d,
          network.neuron_array(neuron_index).spike_function()
        );
      }break;
      case ad_operation_neuron_spike_d:{
        dependencies.push_back(operations.size());
        dependencies.push_back(operations.size() + 1u);
        operations.emplace_back( /* current value provided by the transfer function */
          network, operations, operation_index, ad_operation_neuron_transfer_d,
          network.neuron_array(neuron_index).transfer_function()
        );
        operations.emplace_back( /* previous value of the neuron */
          network, operations, operation_index, ad_operation_neuron_spike_d,
          network.neuron_array(neuron_index).transfer_function(), 1u /*past_index*/
        );
      }break;
      case ad_operation_neuron_transfer_d:{
        dependencies.push_back(operations.size());
        operations.emplace_back(
          network, operations, operation_index, ad_operation_neuron_input_d,
          network.neuron_array(operation_index).input_function_function(), 0u /*neuron_input_index*/,
          network.weight_table(network.neuron_array(operation_index).input_weights(0).interval_start()) /* the weight */
        );
      }break;
      case ad_operation_neuron_input_d:{
        if(){ /* current operation is calculating a value */

        }
        std::uint32_t neuron_input_index = std::get<1>(type_arguments);
        std::uint32_t weight_index_in_input_synapse = 1u + neuron_input_index; /* first index is the spike index */
        rafko_net::SynapseIterator<InputSynapseInterval> inputs_iterator(network.neuron_array(operation_index).input_indices());
        rafko_net::SynapseIterator<IndexSynapseInterval> weights_iterator(network.neuron_array(operation_index).input_weights());

        if(neuron_input_index < inputs_iterator.cached_size()){ /* push in the other inputs as dependency */
          dependencies.push_back(operations.size());
          operations.emplace_back( /* push in dependency for the value */
            network, operations, operation_index, ad_operation_neuron_input_d,
            network.neuron_array(operation_index).input_function_function(), (neuron_input_index + 1) /*neuron_input_index*/
          );
          dependencies.push_back(operations.size());
          operations.emplace_back( /* push in dependency for the derivate */
            network, operations, operation_index, ad_operation_neuron_input_d,
            network.neuron_array(operation_index).input_function_function(), (neuron_input_index + 1) /*neuron_input_index*/
          );
        }else if(rafko_net::SynapseIterator::is_index_input(inputs_iterator[neuron_input_index])){ /* push in the network input as dependency */
          dependencies.push_back(operations.size());
          operations.emplace_back(
            network, operations, rafko_net::SynapseIterator::synapse_index_from_input_index(inputs_iterator[neuron_input_index]),
            ad_operation_network_input_d
          );
        }else{ /* push in the neuron data as dependency */

        }

        /*!Note: current operation is to calculate the inputs starting from the current index,
         * but the elements starting from the next input is present in the operations vector as a dependency.
         */

        // inputs_iterator.iterate(network.neuron_array(operation_index).input_indices(),
        // [this](std::int32 input_index){
        //   if(rafko_net::SynapseIterator::is_index_input(input_index)){
        //     dependencies.push_back(operations.size());
        //     operations.emplace_back(
        //       network, operations, rafko_net::SynapseIterator::synapse_index_from_input_index(input_index),
        //       ad_operation_network_input_d, weight_index_in_input_synapse/* weight_index */
        //     );
        //   }else{
        //     dependencies.push_back(operations.size());
        //     operations.emplace_back(
        //       network, operations, input_index, ad_operation_neuron_spike_d,
        //       network.neuron_array(neuron_index).spike_function()
        //     );
        //   }
        // });
      }break;
      case ad_operation_network_input_d: /* network inputs should have no dependence */
      default: break;
    }/*switch(operation)*/
    registered = true;
  }

  void calculate(
    std::uint32 d_w_index, std::uint32 run_index,
    const std::vector<double>& error_data, const std::vector<double>& label_data,
    const DataRingbuffer& neuron_data, const std::vector<double>& network_input,
    const std::vector<double>& spike_function_input, const std::vector<double>& transfer_function_input
  ){
    switch(operation){
      case ad_operation_objective_d:{
        RFASSERT(1u == dependencies.size());
        std::uint32_t neuron_index = (network.neuron_array_size() - network.output_neuron_number() + operation_index);
        derivative_value = ( /* Objective E(x,f(x))/dx = E'(x,f(x))/df(x)*f'(x)/dx  */
          std::get<0>(type_arguments).get_derivative( label_data[operation_index], neuron_data[neuron_index]) * dependencies[0]()
        );
      }break;
      case ad_operation_neuron_spike_d:{
        RFASSERT(2u == dependencies.size());
        double spike_derivative = SpikeFunction::get_derivative(..);
        derivative_value = ( /* Spike function S(x,f(x),g(x))/dx = S'x() + g'(x) * S'g() + f'(x) * S'f() */
          spike_derivative + dependencies[0]() * spike_derivative + dependencies[1]() * spike_derivative
        );
      }break;
      case ad_operation_neuron_transfer_d:{
        //TODO
      }break;
      case ad_operation_neuron_input_d:{
        /* dependencies all contain the input value derivatives */

        //Get all input values
      }break;
      case ad_operation_network_input_d:{
        std::uint32_t weight_index = std::get<0>(type_arguments);
        if(d_w_index == weight_index)
          derivative_value = network_input[operation_index];
          else derivative_value = 0.0;
      }break;
    }
    processed = true;
  }

};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_NETWORK_INPUT_OPERATION_H */

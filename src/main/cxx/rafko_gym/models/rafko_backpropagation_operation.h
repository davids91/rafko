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

#ifndef RAFKO_BACKPROPAGATION_OPERATION_H
#define RAFKO_BACKPROPAGATION_OPERATION_H

#include "rafko_global.h"

#include <vector>
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
template<typename ...OperationTypes>
class RAFKO_FULL_EXPORT RafkoBackpropagationOperation{
public:
  RafkoBackpropagationOperation(
    rafko_net::RafkoNet& network_, std::vector<RafkoBackpropagationOperation>& operations_,
    std::uint32_t index, Autodiff_operations operation_, OperationType types_...
  ):operation(operation_)
  , type_arguments(types_...)
  , operation_index(index)
  , network(network_)
  , operations(operations_)
  {
  }

  //TODO: Make threadsafe with mutex maybe?
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
          network.neuron_array(operation_index).input_function_function()
        );
      }break;
      case ad_operation_neuron_input_d:{
        rafko_net::SynapseIterator::iterate(network.neuron_array(operation_index).input_indices(),
        [this](std::int32 input_index){
          if(rafko_net::SynapseIterator::is_index_input(input_index)){
            dependencies.push_back(operations.size());
            operations.emplace_back(
              network, operations, rafko_net::SynapseIterator::synapse_index_from_input_index(input_index),
              ad_operation_neuron_input_d
            );
          }else{
            dependencies.push_back(operations.size());
            operations.emplace_back(
              network, operations, input_index, ad_operation_neuron_spike_d,
              network.neuron_array(neuron_index).spike_function()
            );
          }
        });
      }break;
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
        value = ( /* Objective E(x,f(x))/dx = E'(x,f(x))/df(x)*f'(x)/dx  */
          std::get<0>(type_arguments).get_derivative( label_data[operation_index], neuron_data[neuron_index]) * dependencies[0]()
        );
      }break;
      case ad_operation_neuron_spike_d:{
        RFASSERT(2u == dependencies.size());
        double spike_derivative = SpikeFunction::get_derivative(..);
        value = ( /* Spike function S(x,f(x),g(x))/dx = S'x() + g'(x) * S'g() + f'(x) * S'f() */
          spike_derivative + dependencies[0]() * spike_derivative + dependencies[1]() * spike_derivative
        );
      }break;
      case ad_operation_neuron_transfer_d:{
        //TODO
      }break;
      case ad_operation_neuron_input_d:{
        //TODO
      }break;
    }
    processed = true;
  }

  // std::string get_kernel_function(){
  //
  // }

  double operator()(){
    if(!processed)calculate();
    return value;
  }

  bool constexpr is_registered(){
    return registered;
  }

  bool constexpr is_processed(){
    return processed;
  }

private:
  const Autodiff_operations operation;
  const std::tuple<OperationTypes...> type_arguments;
  const std::uint32_t operation_index;
  rafko_net::RafkoNet& network;
  std::vector<RafkoBackpropagationOperation>& operations;
  std::vector<std::uint32_t> dependencies;

  bool processed = false;
  bool registered = false;
  double value = 0.0;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROPAGATION_OPERATION_H */

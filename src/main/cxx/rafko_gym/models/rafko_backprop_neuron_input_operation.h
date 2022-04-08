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

#include "rafko_gym/models/rafko_backpropagation_operation.h"
#include "rafko_gym/models/rafko_backprop_network_input_operation.h"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_FULL_EXPORT RafkoBackpropNeuronInputOperation{
public:
  RafkoBackpropNeuronInputOperation(
    RafkoBackPropagation& queue, rafko_net::RafkoNet& network,
    std::uint32_t neuron_index_, std::uint32_t neuron_input_index_,
    std::shared_ptr<double> tmp_value_buffer_ = {}
  ):RafkoBackpropagationOperation(queue, network)
  , neuron_index(neuron_index_)
  , neuron_input_index(neuron_input_index_)
  , tmp_value_buffer(tmp_value_buffer_)
  , inputs_iterator(network.neuron_array(neuron_index).input_indices())
  , weights_iterator(network.neuron_array(neuron_index).input_weights())
  , is_network_input(rafko_net::SynapseIterator::is_index_input(inputs_iterator[neuron_input_index]))
  , input_index_from_neuron_input_index(
    (!is_network_input)?(inputs_iterator[neuron_input_index])
    :(rafko_net::SynapseIterator::input_index_to_synapse_index(inputs_iterator[neuron_input_index]))
  ),weight_index(weights_iterator[1u + neuron_input_index] /* spike index preceeds the inputs */
  {
  }

  void upload_dependencies_to_operations(){
    if(0 == operation_index)tmp_value_buffer = std::make_shared<double>();
    /* push in dependency for current input derivative */
    if(is_network_input){
      network_input_dependency = queue.push_dependency(ad_operation_network_input_d,
        queue, network, input_index_from_neuron_input_index, weights_iterator[1 + neuron_input_index]
      );
    }else{ /* if it's not an input, then it's an internal neuron value */
      neuron_data_dependency = queue.push_dependency(ad_operation_neuron_spike_d,
        network, queue, input_index_from_neuron_input_index, operations
      );
    }
    if(neuron_input_index < (inputs_iterator.cached_size() - 1u)){
      neuron_input_dependency = queue.push_dependency(
        ad_operation_neuron_input_d, queue, network,
        neuron_index, (neuron_input_index + 1), tmp_value_buffer, operations
      ); /* push in dependency u(x) = every input after this one */
    }
    /*!Note: current operation is to calculate the inputs starting from the current index,
     * but the elements starting from the next input is present in the operations vector as a dependency.
     */
    set_registered();
  }

  void calculate(
    std::uint32 d_w_index, std::uint32 run_index,
    const std::vector<double>& error_data, const std::vector<double>& label_data,
    const DataRingbuffer& neuron_data, const std::vector<double>& network_input,
    const std::vector<double>& spike_function_input
  ){
    RFASSERT(are_dependencies_registered());
    if(neuron_input_index < (inputs_iterator.cached_size() - 1u)){ /* not the last input */
      /* i(w) = w * f(w) ¤ u(w) | f(w) = network_input or input_from_internal_neuron */
      RFASSERT(neuron_input_dependency);
      RFASSERT(neuron_input_dependency->is_processed());
      double weighted_input;
      std::shared_ptr<RafkoBackpropagationOperation> derivative_ptr;
      if(is_network_input){
        RFASSERT(network_input_dependency);
        RFASSERT(network_input_dependency->is_processed());
        weighted_input = network_input[input_index_from_neuron_input_index];
        derivative_ptr = network_input_dependency;
      }else{
        RFASSERT(neuron_data_dependency);
        RFASSERT(neuron_data_dependency->is_processed());
        weighted_input = neuron_data[input_index_from_neuron_input_index];
        derivative_ptr = neuron_data_dependency;
      }
      weighted_input *= network.weight_table(weight_index);
      derivative_value = InputFunction::get_derivative(
        network.neuron_array(neuron_index).input_function(),
        weighted_input, derivative_ptr->get_derivative(),
        tmp_value_buffer, neuron_input_dependency->get_derivative()
      );
      tmp_value_buffer = InputFunction::collect(
        network.neuron_array(neuron_index).input_function(), weighted_input, tmp_value_buffer
      );
      value = tmp_value_buffer;
    }else{ /* the last input: i(w) = w * f(w) */
      std::shared_ptr<RafkoBackpropagationOperation> derivative_ptr;
      if(is_network_input){
        RFASSERT(network_input_dependency);
        RFASSERT(network_input_dependency->is_processed());
        tmp_value_buffer = network_input[input_index_from_neuron_input_index];
        derivative_ptr = network_input_dependency;
      }else{
        RFASSERT(neuron_data_dependency);
        RFASSERT(neuron_data_dependency->is_processed());
        tmp_value_buffer = neuron_data[input_index_from_neuron_input_index];
        derivative_ptr = neuron_data_dependency;
      }
      tmp_value_buffer *= network.weight_table(weight_index);
      derivative_value = ( /* d i(w)/dw = w * f'(w) + f(w) */
        network.weight_table(weight_index) * derivative_ptr->get_derivate()
        + tmp_value_buffer
      );
      value = tmp_value_buffer;
    }
    set_processed();
  }

private:
  std::uint32_t neuron_index;
  std::uint32_t neuron_input_index;
  std::shared_ptr<double> tmp_value_buffer; /* common buffer for partial results of each input */
  rafko_net::SynapseIterator<InputSynapseInterval> inputs_iterator;
  rafko_net::SynapseIterator<IndexSynapseInterval> weights_iterator;

  const std::uint32_t input_index_from_neuron_input_index;
  const std::uint32_t weight_index;
  const bool is_network_input;

  std::shared_ptr<RafkoBackpropagationOperation> network_input_dependency;
  std::shared_ptr<RafkoBackpropagationOperation> neuron_data_dependency;
  std::shared_ptr<RafkoBackpropagationOperation> neuron_input_dependency;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_NEURON_INPUT_OPERATION_H */

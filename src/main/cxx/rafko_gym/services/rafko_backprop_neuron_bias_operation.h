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

#ifndef RAFKO_BACKPROP_NEURON_BIAS_OPERATION_H
#define RAFKO_BACKPROP_NEURON_BIAS_OPERATION_H

#include "rafko_global.h"

#include <vector>
#include <memory>
#include <utility>
#if(RAFKO_USES_OPENCL)
#include <string>
#endif/*(RAFKO_USES_OPENCL)*/

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/services/rafko_assertion_logger.h"

#include "rafko_gym/services/rafko_backpropagation_operation.h"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_FULL_EXPORT RafkoBackpropNeuronBiasOperation
: public RafkoBackpropagationOperation
{
public:
  RafkoBackpropNeuronBiasOperation(
    RafkoBackPropagationData& data, const rafko_net::RafkoNet& network,
    std::uint32_t operation_index, std::uint32_t neuron_index_, std::uint32_t neuron_weight_index_
  )
  : RafkoBackpropagationOperation(data, network, operation_index)
  , neuron_index(neuron_index_)
  , neuron_weight_index(neuron_weight_index_)
  , weights_iterator(network.neuron_array(neuron_index).input_weights())
  , weight_index(weights_iterator[neuron_weight_index])
  {
  }

  DependencyRequest upload_dependencies_to_operations(){
    if(neuron_weight_index < (weights_iterator.cached_size() - 1u)){ /* more biases are present with the Neuron */
      return {{
        {{ad_operation_neuron_bias_d,{neuron_index, (neuron_weight_index + 1u)}}},
        [this](std::vector<std::shared_ptr<RafkoBackpropagationOperation>> dependencies){
          RFASSERT(1u == dependencies.size());
          next_bias_dependency = dependencies[0];
          set_registered();
        }
      }};
    }else{
      set_registered();
      return {};
    }
  }

  void calculate(
    std::uint32_t d_w_index, std::uint32_t run_index,
    const std::vector<std::vector<double>>& network_input,
    const std::vector<std::vector<double>>& label_data
  ){
    RFASSERT(run_index < network_input.size());
    RFASSERT(run_index < label_data.size());
    double dependency_value = 0.0;
    double dependency_derivative = 0.0;
    if(neuron_weight_index < (weights_iterator.cached_size() - 1u)){
      RFASSERT(static_cast<bool>(next_bias_dependency));
      RFASSERT(next_bias_dependency->is_processed());
      dependency_value = next_bias_dependency->get_value(run_index);
      dependency_derivative = next_bias_dependency->get_derivative(run_index, d_w_index);
      set_value(run_index, rafko_net::InputFunction::collect(
        network.neuron_array(neuron_index).input_function(),
        network.weight_table(weight_index), dependency_value
      ));
      set_derivative(run_index, d_w_index, rafko_net::InputFunction::get_derivative(
        network.neuron_array(neuron_index).input_function(),
        network.weight_table(weight_index), ((d_w_index == weight_index)?(1.0):(0.0)),
        dependency_value, dependency_derivative
      ));
    }else{ /* no additional bias values are present as dependencies */
      set_value( run_index, network.weight_table(weight_index) );
      set_derivative( run_index, d_w_index, ((d_w_index == weight_index)?(1.0):(0.0)) );

    }

    set_processed();
  }

  #if(RAFKO_USES_OPENCL)
  std::string value_kernel_function() const{
    std::string next_dependency_string = "";
    if(neuron_weight_index < (weights_iterator.cached_size() - 1u)){
      RFASSERT(static_cast<bool>(next_bias_dependency));
      RFASSERT(next_bias_dependency->are_dependencies_registered());
      next_dependency_string = " -+-> " + next_bias_dependency->value_kernel_function();
    }
    return (
      "|| \t ---> bias weight[" + std::to_string(weight_index) + "]"
      + "(" + std::to_string(network.weight_table(weight_index)) + ")"
      + next_dependency_string
    );
  }
  std::string derivative_kernel_function() const{
    return "";
  }
  #endif/*(RAFKO_USES_OPENCL)*/

  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> get_dependencies(){
    return {};
  }

private:
  const std::uint32_t neuron_index;
  const std::uint32_t neuron_weight_index;
  rafko_net::SynapseIterator<rafko_net::IndexSynapseInterval> weights_iterator;
  const std::uint32_t weight_index;

  std::shared_ptr<RafkoBackpropagationOperation> next_bias_dependency;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_NEURON_BIAS_OPERATION_H */

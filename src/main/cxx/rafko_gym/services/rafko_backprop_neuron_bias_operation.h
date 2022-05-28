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
#include <regex>
#endif/*(RAFKO_USES_OPENCL)*/

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/services/rafko_assertion_logger.h"
#include "rafko_net/services/synapse_iterator.h"
#include "rafko_net/models/input_function.h"

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
    RafkoBackpropagationData& data, const rafko_net::RafkoNet& network,
    std::uint32_t operation_index, std::uint32_t neuron_index_, std::uint32_t neuron_weight_index_
  )
  : RafkoBackpropagationOperation(data, network, operation_index, ad_operation_neuron_bias_d)
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

  void calculate_value(const std::vector<double>& network_input);

  void calculate_derivative(
    std::uint32_t d_w_index, const std::vector<double>& network_input, const std::vector<double>& label_data
  );

  #if(RAFKO_USES_OPENCL)
  std::string value_kernel_operation(
    std::string /*network_input_array*/, std::string /*network_input_array_start*/,
    std::string weight_array, std::string weight_array_start,
    std::string operations_value_array, std::string operations_value_array_start,
    std::string /*operations_array_size*/
  ) const;

  std::string derivative_kernel_operation(
    std::string /*network_input_array*/, std::string /*network_input_array_start*/,
    std::string /*label_array*/, std::string /*label_array_start*/,
    std::string weight_array, std::string weight_array_start,
    std::string operations_value_array, std::string operations_value_array_start,
    std::string operations_derivative_array, std::string operations_derivative_array_start,
    std::string /*operations_array_size*/
  ) const;
  #endif/*(RAFKO_USES_OPENCL)*/

  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> get_dependencies(){
    if(neuron_weight_index < (weights_iterator.cached_size() - 1u)){
      return {next_bias_dependency};
    }else return {};
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

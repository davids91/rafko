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
    RafkoBackpropagationData& data, const rafko_net::RafkoNet& network,
    std::uint32_t operation_index, std::uint32_t neuron_index_, std::uint32_t neuron_input_index_
  );
  ~RafkoBackpropNeuronInputOperation() = default;

  DependencyRequest upload_dependencies_to_operations();

  void calculate_value(const std::vector<double>& network_input);
  void calculate_derivative(
    std::uint32_t d_w_index, const std::vector<double>& network_input, const std::vector<double>& label_data
  );

  #if(RAFKO_USES_OPENCL)
  std::string local_declaration_operation() const;

  std::string value_kernel_operation(
    std::string network_input_array, std::string weight_array,
    std::string operations_value_array, std::string operations_array_size
  ) const;
  std::string derivative_kernel_operation(
    std::string network_input_array, std::string label_array, std::string weight_array,
    std::string operations_value_array, std::string operations_derivative_array,
    std::string operations_array_size
  ) const;
  #endif/*(RAFKO_USES_OPENCL)*/

  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> get_dependencies();

private:
  const std::uint32_t neuron_index;
  const std::uint32_t neuron_input_index;
  rafko_net::SynapseIterator<rafko_net::InputSynapseInterval> inputs_iterator;
  rafko_net::SynapseIterator<rafko_net::IndexSynapseInterval> weights_iterator;

  const bool is_network_input;
  const std::uint32_t input_past_index;
  const std::uint32_t weight_index;

  std::shared_ptr<RafkoBackpropagationOperation> network_input_dependency;
  std::shared_ptr<RafkoBackpropagationOperation> neuron_data_dependency;
  std::shared_ptr<RafkoBackpropagationOperation> neuron_input_dependency;
  std::shared_ptr<RafkoBackpropagationOperation> neuron_bias_dependency;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_NEURON_INPUT_OPERATION_H */

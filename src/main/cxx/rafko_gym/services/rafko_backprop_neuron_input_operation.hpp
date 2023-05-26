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

#include "rafko_global.hpp"

#include <vector>
#include <memory>
#include <utility>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/services/rafko_assertion_logger.hpp"
#include "rafko_net/models/input_function.hpp"
#include "rafko_net/services/synapse_iterator.hpp"

#include "rafko_gym/services/rafko_backpropagation_operation.hpp"
#include "rafko_gym/services/rafko_backprop_network_input_operation.hpp"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_EXPORT RafkoBackpropNeuronInputOperation
: public RafkoBackpropagationOperation
{
public:
  RafkoBackpropNeuronInputOperation(
    RafkoBackpropagationData& data, const rafko_net::RafkoNet& network,
    std::uint32_t operation_index, std::uint32_t neuron_index, std::uint32_t neuron_input_index
  );
  ~RafkoBackpropNeuronInputOperation() = default;

  std::uint32_t get_weight_index() const{
    return m_weightIndex;
  }

  std::uint32_t get_input_past_index() const{
    return m_inputPastIndex;
  }

  rafko_net::Input_functions get_input_function() const{
    return m_network.neuron_array(m_neuronIndex).input_function();
  }

  /**
   * @brief     Provides all the dependency pointers, irregardless if the inputs are from the past or not.
   *            The distinction is relevant, because past inputs are not defining operation structures, 
   *            but for inference index values they are required.
   * 
   * @return    list of all stored dependency pointers
   */
  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> get_own_dependencies_past_included();

  DependencyRequest upload_dependencies_to_operations() override;

  void calculate_value(const std::vector<double>& network_input) override;
  void calculate_derivative(
    std::uint32_t d_w_index, const std::vector<double>& network_input, const std::vector<double>& label_data
  ) override;

  #if(RAFKO_USES_OPENCL)
  std::string local_declaration_operation() const override;


  /**
   * @brief     Generates OpenCL Kernel code for the operation for forward propagation
   * 
   * @param   weight_array                  The name of the array contining the Neural network weights 
   * @param   operations_value_array        The name of the array containing the operation values for forward propagation
   * @param   operations_array_size         The size of the array contining the operation values for both forward and backward propagation
   *
   * @return    Raw Kernel code for the forward propagation of this operation
   */
  static std::string generic_value_kernel_operation(
    std::string weight_array, std::string operations_value_array, std::string operations_array_size,
    std::string behavior_index, std::string past_index, std::string weight_is_used
  );

  std::string value_kernel_operation(
    std::string network_input_array, std::string weight_array,
    std::string operations_value_array, std::string operations_array_size
  ) const override;

  /**
   * @brief     Generates OpenCL Kernel code for the operation for backward propagation
   * 
   * @param   weight_array                  The name of the array contining the Neural network weights 
   * @param   operations_value_array        The name of the array containing the operation values for backward propagation
   * @param   operations_derivative_array   The name of the array containing the operation values for backward propagation
   * @param   operations_array_size         The size of the array contining the operation values for both forward and backward propagation
   * @param   behavior_index                The value determining the input function of the operation
   * @param   past_index                    The value determining the past index inside this operations input 
   *
   * @return    Raw Kernel code for the forward propagation of this operation
   */
  static std::string generic_derivative_kernel_operation(
    std::string weight_array, std::string operations_value_array, std::string operations_derivative_array,
    std::string operations_array_size, std::string behavior_index, std::string past_index
  );

  std::string derivative_kernel_operation(
    std::string network_input_array, std::string label_array, std::string weight_array,
    std::string operations_value_array, std::string operations_derivative_array,
    std::string operations_array_size
  ) const override;
  void substitute_index_values_in_kernels(std::string& kernel_source) const override;
  #endif/*(RAFKO_USES_OPENCL)*/

  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> get_own_dependencies() override;

private:
  const std::uint32_t m_neuronIndex;
  const std::uint32_t m_neuronInputIndex;
  rafko_net::SynapseIterator<rafko_net::InputSynapseInterval> m_inputsIterator;
  rafko_net::SynapseIterator<rafko_net::IndexSynapseInterval> m_weightsIterator;

  const bool m_isNetworkInput;
  const std::uint32_t m_inputPastIndex;
  const std::uint32_t m_weightIndex;

  std::shared_ptr<RafkoBackpropagationOperation> m_networkInputDependency;
  std::shared_ptr<RafkoBackpropagationOperation> m_neuronDataDependency;
  std::shared_ptr<RafkoBackpropagationOperation> m_neuronInputDependency;
  std::shared_ptr<RafkoBackpropagationOperation> m_neuronBiasDependency;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_NEURON_INPUT_OPERATION_H */

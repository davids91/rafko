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

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "rafko_mainframe/services/rafko_assertion_logger.hpp"
#include "rafko_net/models/input_function.hpp"
#include "rafko_net/services/synapse_iterator.hpp"
#include "rafko_protocol/rafko_net.pb.h"

#include "rafko_gym/services/rafko_backpropagation_operation.hpp"

namespace rafko_gym {

/**
 * @brief A backpropagation operation to calculate value and derivative for
 * part of a Neuron input. One operation contains a single input either from
 * the Newtork input or from other (neuron spike) operations.
 */
class RAFKO_EXPORT RafkoBackpropNeuronInputOperation
    : public RafkoBackpropagationOperation {
public:
  using DependencyPointer = std::shared_ptr<RafkoBackpropagationOperation>;

  RafkoBackpropNeuronInputOperation(RafkoBackpropagationData &data,
                                    const rafko_net::RafkoNet &network,
                                    std::uint32_t operation_index,
                                    std::uint32_t neuron_index,
                                    std::uint32_t neuron_input_index);
  ~RafkoBackpropNeuronInputOperation() = default;

  rafko_net::Input_functions get_input_function() const {
    return m_network.neuron_array(m_neuronIndex).input_function();
  }

  /**
   * @brief     Provides all the dependency pointers, irregardless if the inputs
   * are from the past or not. The distinction is relevant, because past inputs
   * are not defining operation structures, but for inference index values they
   * are required.
   *
   * @return    list of all stored dependency pointers
   */
  std::vector<DependencyPointer> get_own_dependencies_past_included();

  DependencyRequest request_dependencies() override;

  void calculate_value(const std::vector<double> &network_input) override;
  void calculate_derivative(std::uint32_t d_w_index,
                            const std::vector<double> &network_input,
                            const std::vector<double> &label_data) override;

#if (RAFKO_USES_OPENCL)

  std::uint32_t get_f_x_dependency_index() const {
    if (m_inputPastIndex.has_value()) {
      RFASSERT(static_cast<bool>(m_neuronDataDependency));
      return m_neuronDataDependency->get_operation_index();
    }
    return m_inputIndex;
  }

  std::uint32_t get_u_x_dependency_index() const {
    if (m_nextDependency) {
      return m_nextDependency->get_operation_index();
    }
    return 0xFFFFFFFFu;
  }

  std::uint8_t get_input_past_index() const {
    if (!m_inputPastIndex.has_value()) {
      return 0xFFu;
    }
    return m_inputPastIndex.value();
  }

  std::string local_declaration_operation() const override;

  /**
   * @brief     Generates OpenCL Kernel code for the operation for forward
   * propagation
   *
   * @param   network_input_array           The name of the array containing the
   * network input values for forward propagation
   * @param   weight_array                  The name of the array contining the
   * Neural network weights
   * @param   operations_value_array        The name of the array containing the
   * operation values for forward propagation
   * @param   operations_array_size         The size of the array contining the
   * operation values for both forward and backward propagation
   * @param   behavior_index                The value determining the input
   * function of the operation
   *
   * @return    Raw Kernel code for the forward propagation of this operation
   */
  static std::string generic_value_kernel_operation(
      std::string network_input_array, std::string weight_array,
      std::string operations_value_array, std::string operations_array_size,
      std::string behavior_index);

  /**
   * @brief     Generates OpenCL Kernel code for the operation for backward
   * propagation
   *
   * @param   network_input_array           The name of the array containing the
   * network input values for forward propagation
   * @param   weight_array                  The name of the array contining the
   * Neural network weights
   * @param   operations_value_array        The name of the array containing the
   * operation values for backward propagation
   * @param   operations_derivative_array   The name of the array containing the
   * operation values for backward propagation
   * @param   operations_array_size         The size of the array contining the
   * operation values for both forward and backward propagation
   * @param   behavior_index                The value determining the input
   * function of the operation
   *
   * @return    Raw Kernel code for the forward propagation of this operation
   */
  static std::string generic_derivative_kernel_operation(
      std::string network_input_array, std::string weight_array,
      std::string operations_value_array,
      std::string operations_derivative_array,
      std::string operations_array_size, std::string behavior_index);
#endif /*(RAFKO_USES_OPENCL)*/

  std::vector<DependencyPointer> get_own_dependencies() override;

  using InputSynapseInterval = rafko_net::InputSynapseInterval;
  const std::uint32_t m_neuronIndex;
  const std::uint32_t m_neuronInputIndex; /* Inside the Neuron input synapse */
  const rafko_net::SynapseIterator<InputSynapseInterval> m_inputsIterator;
  const rafko_net::SynapseIterator<> m_weightsIterator;
  const std::optional<bool> m_isNextDepBias; /* If there is a next dependency */
  const std::optional<std::uint8_t> m_inputPastIndex;
  const std::uint32_t m_inputIndex; /* Network input or input Neuron data */
  const std::uint32_t m_weightIndex;

private:
  DependencyPointer m_neuronDataDependency;
  DependencyPointer m_nextDependency;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_NEURON_INPUT_OPERATION_H */

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
 * @brief A backpropagastion operation to calculate value and derivative for
 * part of a Neuron input. One operation contains multiple inputs either from
 * the Newtork input or from other (neuron spike) operations. Since Neuron
 * inputs consist of both weight and input synapses, one operation need to
 * consider both aspects. Because of this: one operation can span the number of
 * inputs inside both synapses. Meaning if either synapse would be out of bounds
 * with the next input, a new Input operation need to be created as a
 * dependency. This may require that Input operations may not start at the
 * beginning of each synapse so the boundaries need be stored accordingly.
 * Even the first inputs of a Neuron are unaligned, as the first weight of
 * the Neuron is for the spike function, and not the inputs.
 */
class RAFKO_EXPORT RafkoBackpropNeuronInputOperation
    : public RafkoBackpropagationOperation {
  /*!Note: as is in the constructor: input synapse index, weight synapse index,
   * start in input synapse, start in weight synapse */
  struct SynapseSpan {
    const std::uint32_t m_inputSynapseIndex;
    const std::uint32_t m_weightSynapseIndex;
    const std::uint32_t m_startInInputSynapse;
    const std::uint32_t m_startInWeightSynapse;
    const std::uint32_t __workaround_m_isBias = false;
  };
  struct ConstructKit {
    const std::uint32_t m_inputCount;
    const std::uint32_t m_startingInputIndex;
    const std::uint32_t m_startingWeightIndex;
    const std::optional<std::uint32_t> m_inputPastIndex = std::nullopt;
    const std::optional<SynapseSpan> m_nextSpan = std::nullopt;
  };

  RafkoBackpropNeuronInputOperation(RafkoBackpropagationData &data,
                                    const rafko_net::RafkoNet &network,
                                    std::uint32_t operation_index,
                                    std::uint32_t neuron_index,
                                    ConstructKit kit);

public:
  RafkoBackpropNeuronInputOperation(
      RafkoBackpropagationData &data, const rafko_net::RafkoNet &network,
      std::uint32_t operation_index, std::uint32_t neuron_index,
      std::uint32_t input_synapse_index = 0u,
      std::uint32_t weight_synapse_index = 0u,
      std::uint32_t start_inside_input_synapse = 0u,
      std::uint32_t start_inside_weight_synapse = 1u);
  /*!Note: Spike weight preceeds the inputs, so +1 offset is needed for the
   * default weight synapse start */

  ~RafkoBackpropNeuronInputOperation() = default;

  std::uint32_t get_f_x_dependency_index() const { return 0; }

  std::uint32_t get_u_x_dependency_index() const { return 0; }

  /**
   * @brief   Returns the count of data points this operation requires in the
   * backproagation buffer. This might represent spike or transfer function
   * outputs, bias values etc..
   *
   * @retrun  Number of floating point numbers this operation represents
   */
  std::uint32_t get_data_count() const { return 1u; }

  std::uint32_t get_input_past_index() const {
    if (!m_inputPastIndex.has_value()) {
      return 0xFFu;
    }
    return m_inputPastIndex.value();
  }

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
  std::vector<std::shared_ptr<RafkoBackpropagationOperation>>
  get_own_dependencies_past_included();

  DependencyRequest request_dependencies() override;

  void calculate_value(const std::vector<double> &network_input) override;
  void calculate_derivative(std::uint32_t d_w_index,
                            const std::vector<double> &network_input,
                            const std::vector<double> &label_data) override;

#if (RAFKO_USES_OPENCL)
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

  std::vector<std::shared_ptr<RafkoBackpropagationOperation>>
  get_own_dependencies() override;

  const std::uint32_t m_neuronIndex;
  const std::optional<std::uint32_t> m_inputPastIndex;
  const std::uint32_t m_startingInputIndex;
  const std::uint32_t m_inputCount;
  const std::optional<SynapseSpan> m_nextOperation;

public:
  const std::uint32_t m_startingWeightIndex;

private:
  std::vector<std::shared_ptr<RafkoBackpropagationOperation>>
      m_neuronDataDependencies;
  std::shared_ptr<RafkoBackpropagationOperation> m_nextInputDependency;

  static ConstructKit calculate_current_operation_index_values(
      const rafko_net::RafkoNet &network, std::uint32_t neuron_index,
      std::uint32_t input_synapse_index, std::uint32_t weight_synapse_index,
      std::uint32_t start_inside_input_synapse,
      std::uint32_t start_inside_weight_synapse);
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_NEURON_INPUT_OPERATION_H */

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

#include "rafko_global.hpp"

#include <memory>
#include <utility>
#include <vector>
#if (RAFKO_USES_OPENCL)
#include <regex>
#include <string>
#endif /*(RAFKO_USES_OPENCL)*/

#include "rafko_mainframe/services/rafko_assertion_logger.hpp"
#include "rafko_net/models/input_function.hpp"
#include "rafko_net/services/synapse_iterator.hpp"
#include "rafko_protocol/rafko_net.pb.h"

#include "rafko_gym/services/rafko_backpropagation_operation.hpp"

namespace rafko_gym {

/**
 * @brief
 *
 */
class RAFKO_EXPORT RafkoBackpropNeuronBiasOperation
    : public RafkoBackpropagationOperation {
public:
  RafkoBackpropNeuronBiasOperation(RafkoBackpropagationData &data,
                                   const rafko_net::RafkoNet &network,
                                   std::uint32_t operation_index,
                                   std::uint32_t neuron_index,
                                   std::uint32_t neuron_weight_index)
      : RafkoBackpropagationOperation(data, network, operation_index,
                                      ad_operation_neuron_bias_d),
        m_neuronIndex(neuron_index), m_neuronWeightIndex(neuron_weight_index),
        m_weightsIterator(
            m_network.neuron_array(m_neuronIndex).input_weights()),
        m_weightIndex(m_weightsIterator[m_neuronWeightIndex]) {}
  ~RafkoBackpropNeuronBiasOperation() = default;

  std::uint32_t get_dependency_descriptor() const {
    if (m_nextBiasDependency) {
      return m_nextBiasDependency->get_operation_index();
    }
    return 0xFFFFFFFFu;
  }

  std::uint32_t get_weight_index() const { return m_weightIndex; }

  rafko_net::Input_functions get_input_function() const {
    if (m_nextBiasDependency) {
      return m_network.neuron_array(m_neuronIndex).input_function();
    }
    return rafko_net::Input_functions::input_function_unknown;
  }

  DependencyRequest upload_dependencies_to_operations() override {
    if (m_neuronWeightIndex <
        (m_weightsIterator.cached_size() -
         1u)) { /* more biases are present with the Neuron */
      return {
          {{{ad_operation_neuron_bias_d,
             {m_neuronIndex, (m_neuronWeightIndex + 1u)}}},
           [this](std::vector<std::shared_ptr<RafkoBackpropagationOperation>>
                      dependencies) {
             RFASSERT(1u == dependencies.size());
             m_nextBiasDependency = dependencies[0];
             set_registered();
           }}};
    } else {
      set_registered();
      return {};
    }
  }

  void calculate_value(const std::vector<double> &network_input) override;

  void calculate_derivative(std::uint32_t d_w_index,
                            const std::vector<double> &network_input,
                            const std::vector<double> &label_data) override;

#if (RAFKO_USES_OPENCL)
  std::string local_declaration_operation() const override { return ""; }

  /**
   * @brief     Generates OpenCL Kernel code for the operation for forward
   * propagation
   *
   * @param   weight_array                  The name of the array contining the
   * Neural network weights
   * @param   operations_value_array        The name of the array containing the
   * operation values for forward propagation
   * @param   behavior_index                The index value to determine the
   * Input function in this operation
   *
   * @return    Raw Kernel code for the forward propagation of this operation
   */
  static std::string
  generic_value_kernel_operation(std::string weight_array,
                                 std::string operations_value_array,
                                 std::string behavior_index);

  std::string
  value_kernel_operation(std::string network_input_array,
                         std::string weight_array,
                         std::string operations_value_array,
                         std::string operations_array_size) const override;

  /**
   * @brief     Generates OpenCL Kernel code for the operation for backward
   * propagation
   *
   * @param   weight_array                  The name of the array contining the
   * Neural network weights
   * @param   operations_value_array        The name of the array containing the
   * operation values for backward propagation
   * @param   operations_derivative_array   The name of the array containing the
   * operation values for backward propagation
   * @param   behavior_index                The index to select which input
   * function to use should there be another bias value dependency
   *
   * @return    Raw Kernel code for the backward propagation of this operation
   */
  static std::string generic_derivative_kernel_operation(
      std::string weight_array, std::string operations_value_array,
      std::string operations_derivative_array, std::string behavior_index);
#endif /*(RAFKO_USES_OPENCL)*/

  std::vector<std::shared_ptr<RafkoBackpropagationOperation>>
  get_own_dependencies() override {
    if (m_neuronWeightIndex < (m_weightsIterator.cached_size() - 1u)) {
      RFASSERT(static_cast<bool>(m_nextBiasDependency));
      return {m_nextBiasDependency};
    } else
      return {};
  }

private:
  const std::uint32_t m_neuronIndex;
  const std::uint32_t m_neuronWeightIndex;
  rafko_net::SynapseIterator<rafko_net::IndexSynapseInterval> m_weightsIterator;
  const std::uint32_t m_weightIndex;

  std::shared_ptr<RafkoBackpropagationOperation> m_nextBiasDependency;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_NEURON_BIAS_OPERATION_H */

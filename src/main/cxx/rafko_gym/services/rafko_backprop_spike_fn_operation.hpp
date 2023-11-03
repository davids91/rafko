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

#ifndef RAFKO_BACKPROP_SPIKE_FN_OPERATION_H
#define RAFKO_BACKPROP_SPIKE_FN_OPERATION_H

#include "rafko_global.hpp"

#include <memory>
#include <utility>
#include <vector>

#if (RAFKO_USES_OPENCL)
#include <regex>
#include <string>

#include "rafko_utilities/services/rafko_string_utils.hpp"
#endif /*(RAFKO_USES_OPENCL)*/
#include "rafko_mainframe/services/rafko_assertion_logger.hpp"
#include "rafko_net/models/spike_function.hpp"
#include "rafko_protocol/rafko_net.pb.h"

#include "rafko_gym/services/rafko_backpropagation_operation.hpp"

namespace rafko_gym {

/**
 * @brief
 *
 */
class RAFKO_EXPORT RafkoBackpropSpikeFnOperation
    : public RafkoBackpropagationOperation {
public:
  RafkoBackpropSpikeFnOperation(RafkoBackpropagationData &data,
                                const rafko_net::RafkoNet &network,
                                std::uint32_t operation_index,
                                std::uint32_t neuron_index)
      : RafkoBackpropagationOperation(data, network, operation_index,
                                      ad_operation_neuron_spike_d),
        m_neuronIndex(neuron_index), m_actualOperationIndex(operation_index) {}
  ~RafkoBackpropSpikeFnOperation() = default;

  rafko_net::Spike_functions get_spike_function() const {
    return m_network.neuron_array(m_neuronIndex).spike_function();
  }

  std::uint32_t get_weight_index() const {
    return m_network.neuron_array(m_neuronIndex).input_weights(0).starts();
  }

  std::uint32_t get_operation_index() const override {
    return m_actualOperationIndex;
  }

  void set_operation_index(std::uint32_t index) {
    RFASSERT(!m_operationIndexFinal);
    m_actualOperationIndex = index;
  }

  bool operation_index_finalised() override { return m_operationIndexFinal; }

  DependencyRequest upload_dependencies_to_operations() override {
    return {{{{ad_operation_neuron_transfer_d, {m_neuronIndex}}},
             [this](std::vector<std::shared_ptr<RafkoBackpropagationOperation>>
                        dependencies) {
               RFASSERT(1 == dependencies.size());
               m_presentValueDependency = dependencies[0];
               set_registered();
             }}};
  }

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
   * @param   weight_array                  The name of the array contining the
   * Neural network weights
   * @param   operations_value_array        The name of the array containing the
   * operation values for forward propagation
   * @param   operations_array_size         The size of the array contining the
   * operation values for both forward and backward propagation
   * @param   behavior_index                The index value to determine the
   * Spike function in this operation
   *
   * @return    Raw Kernel code for the forward propagation of this operation
   */
  static std::string generic_value_kernel_operation(
      std::string weight_array, std::string operations_value_array,
      std::string operations_array_size, std::string behavior_index);

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
   * @param   operations_array_size         The size of the array contining the
   * operation values for both forward and backward propagation
   * @param   behavior_index                The index value to determine the
   * Spike function in this operation
   *
   * @return    Raw Kernel code for the backward propagation of this operation
   */
  static std::string generic_derivative_kernel_operation(
      std::string weight_array, std::string operations_value_array,
      std::string operations_derivative_array,
      std::string operations_array_size, std::string behavior_index);
#endif /*(RAFKO_USES_OPENCL)*/

  std::vector<std::shared_ptr<RafkoBackpropagationOperation>>
  get_own_dependencies() override {
    RFASSERT(static_cast<bool>(m_presentValueDependency));
    return {m_presentValueDependency};
  }

private:
  const std::uint32_t m_neuronIndex;
  std::shared_ptr<RafkoBackpropagationOperation> m_presentValueDependency;
  std::uint32_t m_actualOperationIndex;
  bool m_operationIndexFinal = false;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_SPIKE_FN_OPERATION_H */

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

#ifndef RAFKO_BACKPROP_TRANSFER_FN_OPERATION_H
#define RAFKO_BACKPROP_TRANSFER_FN_OPERATION_H

#include "rafko_global.hpp"

#include <memory>
#include <utility>
#include <vector>

#include "rafko_mainframe/models/rafko_settings.hpp"
#include "rafko_mainframe/services/rafko_assertion_logger.hpp"
#include "rafko_net/models/transfer_function.hpp"
#include "rafko_protocol/rafko_net.pb.h"

#include "rafko_gym/services/rafko_backprop_neuron_input_operation.hpp"
#include "rafko_gym/services/rafko_backpropagation_operation.hpp"

namespace rafko_gym {

/**
 * @brief
 *
 */
class RAFKO_EXPORT RafkoBackpropTransferFnOperation
    : public RafkoBackpropagationOperation {
public:
  RafkoBackpropTransferFnOperation(
      RafkoBackpropagationData &data, const rafko_net::RafkoNet &network,
      std::uint32_t operation_index, std::uint32_t neuron_index,
      const rafko_mainframe::RafkoSettings &settings)
      : RafkoBackpropagationOperation(data, network, operation_index,
                                      ad_operation_neuron_transfer_d),
        m_transferFunction(settings), m_neuronIndex(neuron_index) {}
  ~RafkoBackpropTransferFnOperation() = default;

  rafko_net::Transfer_functions get_transfer_function() const {
    return m_network.neuron_array(m_neuronIndex).transfer_function();
  }

  DependencyRequest request_dependencies() override {
    return {{{{ad_operation_neuron_input_d,
               {m_neuronIndex, 0u /*neuron_input_index*/}}},
             [this](std::vector<std::shared_ptr<RafkoBackpropagationOperation>>
                        dependencies) {
               RFASSERT(1 == dependencies.size());
               m_neededInputDependency = dependencies[0];
               set_registered();
             }}};
  }

  void calculate_value(const std::vector<double> & /*network_input*/) override {
    RFASSERT(are_dependencies_registered());
    RFASSERT(static_cast<bool>(m_neededInputDependency));
    RFASSERT(m_neededInputDependency->is_value_processed());
    RFASSERT_LOG("operation[{}]: Neuron[{}] Transfer function = {}({}(op[{}]))",
                 get_operation_index(), m_neuronIndex,
                 Transfer_functions_Name(get_transfer_function()),
                 m_neededInputDependency->get_value(0u /*past_index*/),
                 m_neededInputDependency->get_operation_index());
    set_value(m_transferFunction.get_value(
        get_transfer_function(),
        m_neededInputDependency->get_value(0u /*past_index*/)));
    set_value_processed();
  }

  void calculate_derivative(std::uint32_t d_w_index,
                            const std::vector<double> & /*network_input*/,
                            const std::vector<double> & /*label_data*/
                            ) override {
    RFASSERT(is_value_processed());
    RFASSERT(are_dependencies_registered());
    RFASSERT(static_cast<bool>(m_neededInputDependency));
    RFASSERT(m_neededInputDependency->is_processed());
    set_derivative(d_w_index,
                   m_transferFunction
                       .get_derivative(/* d t(f(w))/dx = f'(w) * t'(f(w))*/
                                       get_transfer_function(),
                                       m_neededInputDependency->get_value(
                                           0u /*past_index*/),
                                       m_neededInputDependency->get_derivative(
                                           0u /*past_index*/, d_w_index)));
    set_derivative_processed();
  }

#if (RAFKO_USES_OPENCL)
  std::string local_declaration_operation() const override { return ""; }

  /**
   * @brief     Generates OpenCL Kernel code for the operation for forward
   * propagation
   *
   * @param   operations_value_array        The name of the array containing the
   * operation values for forward propagation
   *
   * @return    Raw Kernel code for the forward propagation of this operation
   */
  static std::string generic_value_kernel_operation(
      std::string operations_value_array, std::string behavior_index,
      const rafko_mainframe::RafkoSettings &settings) {
    return rafko_net::TransferFunction::get_all_kernel_value_functions(
        settings, behavior_index, operations_value_array + "[==op_index==]",
        operations_value_array + "[==dependency_op_index==]");
  }

  /**
   * @brief     Generates OpenCL Kernel code for the operation for forward
   * propagation
   *
   * @param   network_input_array           The name of the arry containing the
   * Inputs for the Neural network
   * @param   label_array                   The name of the arry containing the
   * Labels the Neural network is evaluated against
   * @param   weight_array                  The name of the array contining the
   * Neural network weights
   * @param   operations_value_array        The name of the array containing the
   * operation values for forward propagation
   * @param   operations_derivative_array   The name of the array containing the
   * operation values for forward propagation
   * @param   operations_array_size         The size of the array contining the
   * operation values for both forward and backward propagation
   *
   * @return    Raw Kernel code for the forward propagation of this operation
   */
  static std::string generic_derivative_kernel_operation(
      std::string operations_value_array,
      std::string operations_derivative_array, std::string behavior_index,
      const rafko_mainframe::RafkoSettings &settings) {
    return (rafko_net::TransferFunction::get_all_kernel_derivative_functions(
               settings, behavior_index,
               operations_derivative_array + "[==op_index==]",
               operations_value_array + "[==dependency_op_index==]",
               operations_derivative_array + "[==dependency_op_index==]")) +
           ";";
  }
#endif /*(RAFKO_USES_OPENCL)*/

  std::vector<std::shared_ptr<RafkoBackpropagationOperation>>
  get_own_dependencies() override {
    RFASSERT(static_cast<bool>(m_neededInputDependency));
    return {m_neededInputDependency};
  }

private:
  const rafko_net::TransferFunction m_transferFunction;
  const std::uint32_t m_neuronIndex;
  std::shared_ptr<RafkoBackpropagationOperation> m_neededInputDependency;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_TRANSFER_FN_OPERATION_H */

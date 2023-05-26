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

#ifndef RAFKO_BACKPROP_OBJECTIVE_OPERATION_H
#define RAFKO_BACKPROP_OBJECTIVE_OPERATION_H

#include "rafko_global.hpp"

#include <vector>
#include <memory>
#include <utility>

#include "rafko_utilities/services/rafko_string_utils.hpp"
#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/services/rafko_assertion_logger.hpp"
#include "rafko_gym/models/rafko_objective.hpp"

#include "rafko_gym/services/rafko_backpropagation_operation.hpp"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_EXPORT RafkoBackpropObjectiveOperation
: public RafkoBackpropagationOperation
{
public:
  RafkoBackpropObjectiveOperation(
    RafkoBackpropagationData& data, const rafko_net::RafkoNet& network,
    const RafkoObjective& objective, std::uint32_t operation_index,
    std::uint32_t output_index,  std::uint32_t sample_number
  )
  : RafkoBackpropagationOperation(data, network, operation_index, ad_operation_objective_d)
  , m_objective(objective)
  , m_outputIndex(output_index)
  , m_sampleNumber(sample_number)
  {
  }

  ~RafkoBackpropObjectiveOperation() = default;

  DependencyRequest upload_dependencies_to_operations() override{
    return {{
      {{ad_operation_neuron_spike_d, {(m_network.neuron_array_size() - m_network.output_neuron_number() + m_outputIndex)}}},
      [this](std::vector<std::shared_ptr<RafkoBackpropagationOperation>> dependencies){
        RFASSERT(1 == dependencies.size());
        m_featureDependency = dependencies[0];
        set_registered();
      }
    }};
  }

  void calculate_value(const std::vector<double>& /*network_input*/) override{
    RFASSERT(are_dependencies_registered());
    RFASSERT(static_cast<bool>(m_featureDependency));
    RFASSERT(m_featureDependency->is_value_processed());
    /*!Note: Value is not being calculated, because they are not of use (as of now.. ) */
    set_value_processed();
  }

  void calculate_derivative(
    std::uint32_t d_w_index, const std::vector<double>& /*network_input*/, const std::vector<double>& label_data
  ) override{
    RFASSERT(is_value_processed());
    RFASSERT(are_dependencies_registered());
    RFASSERT(m_outputIndex < label_data.size());
    RFASSERT(static_cast<bool>(m_featureDependency));
    RFASSERT(m_featureDependency->is_processed());
    set_derivative(d_w_index, m_objective.get_derivative(
      label_data[m_outputIndex], m_featureDependency->get_value(0u/*past_index*/),
      m_featureDependency->get_derivative(0u/*past_index*/, d_w_index), static_cast<double>(m_sampleNumber)
    ));
    RFASSERT_LOG(
      "derivative operation[{}](w[{}]): Objective[{}]_d = {} = derivative({}(label[{}]),{}(op[{}]),{}(d_op),{}(samples))",
      get_operation_index(), d_w_index, m_outputIndex, get_derivative(0u/*past_index*/, d_w_index),
      label_data[m_outputIndex], m_outputIndex,
      m_featureDependency->get_value(0u/*past_index*/), m_featureDependency->get_operation_index(),
      m_featureDependency->get_derivative(0u/*past_index*/, d_w_index), static_cast<double>(m_sampleNumber)
    );
    set_derivative_processed();
  }

  #if(RAFKO_USES_OPENCL)
  std::string local_declaration_operation() const override{
    return "";
  }

  std::string value_kernel_operation(
    std::string /*network_input_array*/, std::string /*weight_array*/,
    std::string /*operations_value_array*/, std::string /*operations_array_size*/
  ) const override{ /*!Note: Value is not being calculated, because they are not of use (as of now.. ) */
    return "";
  }

  /**
   * @brief     Generates OpenCL Kernel code for the operation for backward propagation
   * 
   * @param   label_array                   The name of the arry containing the Labels the Neural network is evaluated against
   * @param   operations_value_array        The name of the array containing the operation values for forward propagation
   * @param   operations_derivative_array   The name of the array containing the operation values for forward propagation
   * @param   sample_number                 The number of samples the dataset contains in the mini-batch
   * @param   objective                     The source of the Kernel code for thi Operation
   *
   * @return    Raw Kernel code for the backward propagation of this operation
   */
  static std::string generic_derivative_kernel_operation(
    std::string label_array, std::string operations_value_array, std::string operations_derivative_array,
    std::string sample_number, const RafkoObjective& objective
  ){
    return (
      operations_derivative_array + "[==op_index==] = "
      + objective.get_derivative_kernel_source(
        label_array + "[==label_index==]",
        operations_value_array + "[==dependency_op_index==]",
        operations_derivative_array + "[==dependency_op_index==]",
        sample_number
      ) + ";"
    );
  }

  std::string derivative_kernel_operation(
    std::string /*network_input_array*/, std::string label_array, std::string /*weight_array*/,
    std::string operations_value_array, std::string operations_derivative_array,
    std::string /*operations_array_size*/
  ) const override{
    RFASSERT(static_cast<bool>(m_featureDependency));
    return (
      operations_derivative_array + "[==op_index==] = "
      + m_objective.get_derivative_kernel_source(
        label_array + "[==label_index==]",
        operations_value_array + "[==dependency_op_index==]",
        operations_derivative_array + "[==dependency_op_index==]",
        std::to_string(m_sampleNumber)
      ) + ";"
    );
  }
  void substitute_index_values_in_kernels(std::string& kernel_source) const override { 
    kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==op_index=="), std::to_string(get_operation_index())
    );
    kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==label_index=="), std::to_string(m_outputIndex)
    );
    kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==dependency_op_index=="), std::to_string(m_featureDependency->get_operation_index())
    );
  }
  #endif/*(RAFKO_USES_OPENCL)*/

  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> get_own_dependencies() override{
    RFASSERT(static_cast<bool>(m_featureDependency));
    return {m_featureDependency};
  }

  Cost_functions get_cost_type() const{
    return m_objective.get_cost_type();
  }

  std::uint32_t get_label_index(){
    return m_outputIndex;
  }

private:
  const RafkoObjective& m_objective;
  const std::uint32_t m_outputIndex;
  const std::uint32_t m_sampleNumber;
  std::shared_ptr<RafkoBackpropagationOperation> m_featureDependency;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_OBJECTIVE_OPERATION_H */

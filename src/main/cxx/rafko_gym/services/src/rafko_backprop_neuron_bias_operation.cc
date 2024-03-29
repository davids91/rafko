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
#include "rafko_gym/services/rafko_backprop_neuron_bias_operation.hpp"

#if (RAFKO_USES_OPENCL)
#include <regex>

#include "rafko_utilities/services/rafko_string_utils.hpp"
#endif /*(RAFKO_USES_OPENCL)*/

namespace rafko_gym {

void RafkoBackpropNeuronBiasOperation::calculate_value(
    const std::vector<double> & /*network_input*/) {
  RFASSERT(are_dependencies_registered());
  if (m_neuronWeightIndex < (m_weightsIterator.cached_size() - 1u)) {
    RFASSERT(static_cast<bool>(m_nextBiasDependency));
    RFASSERT(m_nextBiasDependency->is_value_processed());
    set_value(rafko_net::InputFunction::collect(
        m_network.neuron_array(m_neuronIndex).input_function(),
        m_network.weight_table(m_weightIndex),
        m_nextBiasDependency->get_value(0u /*past_index*/)));
  } else { /* no additional bias values are present as dependencies */
    set_value(m_network.weight_table(m_weightIndex));
  }
  set_value_processed();
}

void RafkoBackpropNeuronBiasOperation::calculate_derivative(
    std::uint32_t d_w_index, const std::vector<double> & /*network_input*/,
    const std::vector<double> & /*label_data*/
) {
  RFASSERT(is_value_processed());
  RFASSERT(are_dependencies_registered());
  if (m_neuronWeightIndex < (m_weightsIterator.cached_size() - 1u)) {
    RFASSERT(static_cast<bool>(m_nextBiasDependency));
    RFASSERT(m_nextBiasDependency->is_processed());
    RFASSERT_LOG(
        "derivative_operation[{}](w[{}]): Neuron[{}] bias_d = {}_d({}, {}, {}, "
        "{})",
        get_operation_index(), d_w_index, m_neuronIndex,
        Input_functions_Name(
            m_network.neuron_array(m_neuronIndex).input_function()),
        m_network.weight_table(m_weightIndex),
        ((d_w_index == m_weightIndex) ? (1.0) : (0.0)),
        m_nextBiasDependency->get_value(0u /*past_index*/),
        m_nextBiasDependency->get_derivative(0u /*past_index*/, d_w_index));
    set_derivative(d_w_index,
                   rafko_net::InputFunction::get_derivative(
                       m_network.neuron_array(m_neuronIndex).input_function(),
                       m_network.weight_table(m_weightIndex),
                       ((d_w_index == m_weightIndex) ? (1.0) : (0.0)),
                       m_nextBiasDependency->get_value(0u /*past_index*/),
                       m_nextBiasDependency->get_derivative(0u /*past_index*/,
                                                            d_w_index)));
  } else { /* no additional bias values are present as dependencies */
    RFASSERT_LOG("derivative_operation[{}](w[{}]): Neuron[{}] bias_d = {}",
                 get_operation_index(), d_w_index, m_neuronIndex,
                 ((d_w_index == m_weightIndex) ? (1.0) : (0.0)));
    set_derivative(d_w_index, ((d_w_index == m_weightIndex) ? (1.0) : (0.0)));
  }
  set_derivative_processed();
}

#if (RAFKO_USES_OPENCL)
std::string RafkoBackpropNeuronBiasOperation::generic_value_kernel_operation(
    std::string weight_array, std::string operations_value_array,
    std::string behavior_index) {
  std::string kernel_source = R"(
    if(==dependency_descriptor== != 0xFFFFFFFFu){
      ==input_fnc==
    }else{
       ==op_value_array==[==op_index==] = ==weight_array==[==this_op_weight_index==];
    }
  )";

  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==input_fnc=="),
      rafko_net::InputFunction::get_all_kernel_value_functions(
          behavior_index, "==op_value_array==[==op_index==]",
          "==op_value_array==[==dependency_descriptor==]",
          "==weight_array==[==this_op_weight_index==]"));
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==op_value_array=="), operations_value_array);
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==weight_array=="), weight_array);
  return kernel_source;
}

std::string
RafkoBackpropNeuronBiasOperation::generic_derivative_kernel_operation(
    std::string weight_array, std::string operations_value_array,
    std::string operations_derivative_array, std::string behavior_index) {
  std::string kernel_source = R"(
    if(==dependency_descriptor== != 0xFFFFFFFFu){
       ==input_fnc==
    }else{
       ==op_derivative_array==[==op_index==] = ((d_w_index == ==this_op_weight_index==)?(1.0):(0.0));
    }
  )";

  std::string input_function_source =
      rafko_net::InputFunction::get_all_kernel_derivative_functions(
          behavior_index, "==op_derivative_array==[==op_index==]",
          weight_array + "[==this_op_weight_index==]",
          "((d_w_index == ==this_op_weight_index==)?(1.0):(0.0))",
          "==op_value_array==[==dependency_descriptor==]",
          "==op_derivative_array==[==dependency_descriptor==]");

  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==input_fnc=="), input_function_source);
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==op_value_array=="), operations_value_array);
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==op_derivative_array=="),
      operations_derivative_array);
  return kernel_source;
}
#endif /*(RAFKO_USES_OPENCL)*/

} /* namespace rafko_gym */

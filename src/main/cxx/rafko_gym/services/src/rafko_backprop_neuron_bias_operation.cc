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

#if(RAFKO_USES_OPENCL)
#include <regex>

#include "rafko_utilities/services/rafko_string_utils.hpp"
#endif/*(RAFKO_USES_OPENCL)*/

namespace rafko_gym{

void RafkoBackpropNeuronBiasOperation::calculate_value(const std::vector<double>& /*network_input*/){
  RFASSERT(are_dependencies_registered());
  if(m_neuronWeightIndex < (m_weightsIterator.cached_size() - 1u)){
    RFASSERT(static_cast<bool>(m_nextBiasDependency));
    RFASSERT(m_nextBiasDependency->is_value_processed());
    set_value(rafko_net::InputFunction::collect(
      m_network.neuron_array(m_neuronIndex).input_function(),
      m_network.weight_table(m_weightIndex), m_nextBiasDependency->get_value(0u/*past_index*/)
    ));
  }else{ /* no additional bias values are present as dependencies */
    set_value(m_network.weight_table(m_weightIndex));
  }
  set_value_processed();
}

void RafkoBackpropNeuronBiasOperation::calculate_derivative(
  std::uint32_t d_w_index, const std::vector<double>& /*network_input*/, const std::vector<double>& /*label_data*/
){
  RFASSERT(is_value_processed());
  RFASSERT(are_dependencies_registered());
  if(m_neuronWeightIndex < (m_weightsIterator.cached_size() - 1u)){
    RFASSERT(static_cast<bool>(m_nextBiasDependency));
    RFASSERT(m_nextBiasDependency->is_processed());
    RFASSERT_LOG(
      "derivative_operation[{}](w[{}]): Neuron[{}] bias_d = {}_d({}, {}, {}, {})",
      get_operation_index(), d_w_index, m_neuronIndex,
      Input_functions_Name(m_network.neuron_array(m_neuronIndex).input_function()),
      m_network.weight_table(m_weightIndex), ((d_w_index == m_weightIndex)?(1.0):(0.0)),
      m_nextBiasDependency->get_value(0u/*past_index*/),
      m_nextBiasDependency->get_derivative(0u/*past_index*/, d_w_index)
    );
    set_derivative(d_w_index, rafko_net::InputFunction::get_derivative(
      m_network.neuron_array(m_neuronIndex).input_function(),
      m_network.weight_table(m_weightIndex), ((d_w_index == m_weightIndex)?(1.0):(0.0)),
      m_nextBiasDependency->get_value(0u/*past_index*/),
      m_nextBiasDependency->get_derivative(0u/*past_index*/, d_w_index)
    ));
  }else{ /* no additional bias values are present as dependencies */
    RFASSERT_LOG(
      "derivative_operation[{}](w[{}]): Neuron[{}] bias_d = {}",
      get_operation_index(), d_w_index, m_neuronIndex, ((d_w_index == m_weightIndex)?(1.0):(0.0))
    );
    set_derivative( d_w_index, ((d_w_index == m_weightIndex)?(1.0):(0.0)) );
  }
  set_derivative_processed();
}

#if(RAFKO_USES_OPENCL)
std::string RafkoBackpropNeuronBiasOperation::value_kernel_operation(
  std::string /*network_input_array*/, std::string weight_array,
  std::string operations_value_array, std::string /*operations_array_size*/
) const{
  std::string kernel_source = R"(
    if(==is_there_dependency==){
      ==op_value_array==[==op_index==] = ==input_fnc==;
    }else{
       ==op_value_array==[==op_index==] = ==weight_array==[==this_op_weight_index==];
    }
  )";

  kernel_source = rafko_utilities::replace_all_in_string(
    kernel_source, std::regex("==input_fnc=="), 
    rafko_net::InputFunction::get_kernel_function_for(
      m_network.neuron_array(m_neuronIndex).input_function(),
      "==weight_array==[==this_op_weight_index==]", "==op_value_array==[==value_dep_op_index==]"
    )
  );
  kernel_source = rafko_utilities::replace_all_in_string(
    kernel_source, std::regex("==op_value_array=="), operations_value_array
  );  
  kernel_source = rafko_utilities::replace_all_in_string(
    kernel_source, std::regex("==weight_array=="), weight_array
  );
  return kernel_source;
}

std::string RafkoBackpropNeuronBiasOperation::derivative_kernel_operation(
  std::string /*network_input_array*/, std::string /*label_array*/, std::string weight_array,
  std::string operations_value_array, std::string operations_derivative_array,
  std::string /*operations_array_size*/, std::string /*d_operations_array_size*/
) const{
  RFASSERT(are_dependencies_registered());
  std::string kernel_source = R"(
    if(==is_there_dependency==){
      ==op_derivative_array==[==op_index==] = ==input_fnc==;
    }else{
       ==op_derivative_array==[==op_index==] = ((d_w_index == ==this_op_weight_index==)?(1.0):(0.0));
    }
  )";
  kernel_source = rafko_utilities::replace_all_in_string(
    kernel_source, std::regex("==input_fnc=="), 
    rafko_net::InputFunction::derivative_kernel_for(
      m_network.neuron_array(m_neuronIndex).input_function(),
      weight_array + "[==this_op_weight_index]", "((d_w_index == ==this_op_weight_index==)?(1.0):(0.0))",
      "==op_value_array==[==value_dep_op_index==]", "==op_derivative_array==[==value_dep_op_index==]"
    )
  );
  kernel_source = rafko_utilities::replace_all_in_string(
    kernel_source, std::regex("==op_value_array=="), operations_value_array
  );
  kernel_source = rafko_utilities::replace_all_in_string(
    kernel_source, std::regex("==op_derivative_array=="), operations_derivative_array
  );
  return kernel_source;
}

void RafkoBackpropNeuronBiasOperation::substitute_index_values_in_kernels(std::string& kernel_source) const{ 
  kernel_source = rafko_utilities::replace_all_in_string(
    kernel_source, std::regex("==is_there_dependency=="), 
    (m_neuronWeightIndex < (m_weightsIterator.cached_size() - 1u))?"true":"false"
  );
  kernel_source = rafko_utilities::replace_all_in_string(kernel_source, std::regex("==op_index=="), std::to_string(get_operation_index()));
  kernel_source = rafko_utilities::replace_all_in_string(
    kernel_source, std::regex("==this_op_weight_index=="), std::to_string(m_weightIndex)
  );
  if(m_nextBiasDependency){
    kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==value_dep_op_index=="), std::to_string(m_nextBiasDependency->get_operation_index())
    );    
  }else{
    kernel_source = rafko_utilities::replace_all_in_string(kernel_source, std::regex("==value_dep_op_index=="), "0");
  }
}

#endif/*(RAFKO_USES_OPENCL)*/

} /* namespace rafko_gym */

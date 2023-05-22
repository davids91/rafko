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
#include "rafko_gym/services/rafko_backprop_spike_fn_operation.hpp"

namespace rafko_gym{

void RafkoBackpropSpikeFnOperation::calculate_value(const std::vector<double>& /*network_input*/){
  RFASSERT(are_dependencies_registered());
  RFASSERT(static_cast<bool>(m_presentValueDependency));
  RFASSERT(m_presentValueDependency->is_value_processed());
  double past_value = get_value(1u/*past_index*/);
  set_value(rafko_net::SpikeFunction::get_value(
    m_network.neuron_array(m_neuronIndex).spike_function(),
    m_network.weight_table(m_network.neuron_array(m_neuronIndex).input_weights(0).starts()),
    m_presentValueDependency->get_value(0u/*past_index*/), past_value
  ));
  RFASSERT_LOG(
    "operation[{}]: Neuron[{}] Spike(present:{}(op[{}]),past:{}, weight:{}) = {} (calculated with {})", get_operation_index(), m_neuronIndex,
    m_presentValueDependency->get_value(0u/*past_index*/), m_presentValueDependency->get_operation_index(), past_value,
    m_network.weight_table(m_network.neuron_array(m_neuronIndex).input_weights(0).starts()),
    get_value(0u/*past_index*/), Spike_functions_Name(m_network.neuron_array(m_neuronIndex).spike_function())
  );
  set_value_processed();
}

void RafkoBackpropSpikeFnOperation::calculate_derivative(
  std::uint32_t d_w_index,
  const std::vector<double>& /*network_input*/, const std::vector<double>& /*label_data*/
){
  RFASSERT(is_value_processed());
  RFASSERT(are_dependencies_registered());
  RFASSERT(static_cast<bool>(m_presentValueDependency));
  RFASSERT(m_presentValueDependency->is_processed());
  if(static_cast<std::int32_t>(d_w_index) == m_network.neuron_array(m_neuronIndex).input_weights(0).starts()){
    set_derivative(d_w_index, rafko_net::SpikeFunction::get_derivative_for_w(
      m_network.neuron_array(m_neuronIndex).spike_function(),
      m_network.weight_table(m_network.neuron_array(m_neuronIndex).input_weights(0).starts()),
      m_presentValueDependency->get_value(0u/*past_index*/),
      m_presentValueDependency->get_derivative(0u/*past_index*/, d_w_index),
      get_value(1u/*past_index*/), get_derivative(1u/*past_index*/, d_w_index)
    ));
    RFASSERT_LOG(
      "derivative operation[{}](w[{}]): Neuron[{}] Spike_d for {} = {}_d({}(w[{}]),{}(op[{}]), {}(op_d), {}(past_value), {}(past_derivative))",
      get_operation_index(), d_w_index, m_neuronIndex,
      get_derivative(0u/*past_index*/, d_w_index),
      Spike_functions_Name(m_network.neuron_array(m_neuronIndex).spike_function()),
      m_network.weight_table(m_network.neuron_array(m_neuronIndex).input_weights(0).starts()),
      m_network.neuron_array(m_neuronIndex).input_weights(0).starts(),
      m_presentValueDependency->get_value(0u/*past_index*/), m_presentValueDependency->get_operation_index(),
      m_presentValueDependency->get_derivative(0u/*past_index*/, d_w_index),
      get_value(1u/*past_index*/), get_derivative(1u/*past_index*/, d_w_index)
    );
  }else{
    set_derivative(d_w_index, rafko_net::SpikeFunction::get_derivative_not_for_w(
      m_network.neuron_array(m_neuronIndex).spike_function(),
      m_network.weight_table(m_network.neuron_array(m_neuronIndex).input_weights(0).starts()),
      m_presentValueDependency->get_derivative(0u/*past_index*/, d_w_index), get_derivative(1u/*past_index*/, d_w_index)
    ));
    RFASSERT_LOG(
      "derivative operation[{}](w[{}]): Neuron[{}] Spike_d for {} = {}_d'({}(w[{}]), {}(op_d[{}]), {}(past_derivative))",
      get_operation_index(), d_w_index, m_neuronIndex,
      get_derivative(0u/*past_index*/, d_w_index),
      Spike_functions_Name(m_network.neuron_array(m_neuronIndex).spike_function()),
      m_network.weight_table(m_network.neuron_array(m_neuronIndex).input_weights(0).starts()),
      m_network.neuron_array(m_neuronIndex).input_weights(0).starts(),
      m_presentValueDependency->get_derivative(0u/*past_index*/, d_w_index),
      m_presentValueDependency->get_operation_index(),
      get_derivative(1u/*past_index*/, d_w_index)
    );
  }
  set_derivative_processed();
}

#if(RAFKO_USES_OPENCL)
std::string RafkoBackpropSpikeFnOperation::local_declaration_operation() const{
  return R"( /* Spike Function Operation locals */
    double past_value;
    double past_derivative_value;
  )";
}

std::string RafkoBackpropSpikeFnOperation::value_kernel_operation(
  std::string /*network_input_array*/, std::string weight_array,
  std::string operations_value_array, std::string operations_array_size
) const{
  RFASSERT(are_dependencies_registered());
  RFASSERT(static_cast<bool>(m_presentValueDependency));
  std::string kernel_code = R"(
    if(0 < available_memory_slots){
      past_value = ==op_value_array==[==op_index== - ==op_array_size==];
    }else{
      past_value = 0.0;
    }
    ==op_value_array==[==op_index==] = ==spike_kernel==;
  )";

  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==spike_kernel=="), rafko_net::SpikeFunction::get_kernel_function_for(
      m_network.neuron_array(m_neuronIndex).spike_function(),
      "==op_value_array==[==value_dep_op_index==]", "past_value",
      weight_array + "[" + std::to_string(m_network.neuron_array(m_neuronIndex).input_weights(0).starts()) + "]"
    )
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==op_value_array=="), operations_value_array
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==op_array_size=="), operations_array_size
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==op_index=="), std::to_string(get_operation_index())
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==value_dep_op_index=="), std::to_string(m_presentValueDependency->get_operation_index())
  );
  return kernel_code;
}

std::string RafkoBackpropSpikeFnOperation::derivative_kernel_operation(
  std::string /*network_input_array*/, std::string /*label_array*/, std::string weight_array,
  std::string operations_value_array, std::string operations_derivative_array,
  std::string operations_array_size, std::string d_operations_array_size
) const{
  RFASSERT(are_dependencies_registered());
  RFASSERT(static_cast<bool>(m_presentValueDependency));
  /*!Note: Past values are supposed to be mapped just before the current array, so
   * the negative index should contain the previous run. It the responsibility of the caller
   * to make sure there is no out pf bounds error with these index values.
   */
  std::string kernel_code = R"(
    if(0 < available_memory_slots){
      past_value = ==op_value_array==[==op_index== - ==op_array_size==];
      past_derivative_value = ==op_derivative_array==[==op_index== - ==op_d_array_size==];
    }else{
      past_value = 0.0;
      past_derivative_value = 0.0;
    }
    if(d_w_index == ==this_op_weight_index==){
      ==op_derivative_array==[==op_index==] = ==spike_w_kernel==;
    }else{
      ==op_derivative_array==[==op_index==] = ==spike_kernel==;
    }
  )";
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==spike_w_kernel=="), rafko_net::SpikeFunction::get_derivative_kernel_for_w(
      m_network.neuron_array(m_neuronIndex).spike_function(),
      weight_array + "[" + std::to_string(m_network.neuron_array(m_neuronIndex).input_weights(0).starts()) + "]",
      "==op_value_array==[==value_dep_op_index==]", "==op_derivative_array==[==value_dep_op_index==]",
      "past_value", "past_derivative_value"
    )
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==spike_kernel=="), rafko_net::SpikeFunction::get_derivative_kernel_not_for_w(
      m_network.neuron_array(m_neuronIndex).spike_function(),
      weight_array + "[" + std::to_string(m_network.neuron_array(m_neuronIndex).input_weights(0).starts()) + "]",
      "==op_derivative_array==[==value_dep_op_index==]",
      "past_derivative_value"
    )
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==this_op_weight_index=="),
    std::to_string( m_network.neuron_array(m_neuronIndex).input_weights(0).starts() )
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==op_value_array=="), operations_value_array
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==op_array_size=="), operations_array_size
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==op_d_array_size=="), d_operations_array_size
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==op_derivative_array=="), operations_derivative_array
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==op_index=="), std::to_string(get_operation_index())
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==value_dep_op_index=="), std::to_string(m_presentValueDependency->get_operation_index())
  );
  return kernel_code;
}
#endif/*(RAFKO_USES_OPENCL)*/

} /* namespace rafko_gym */

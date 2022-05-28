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
#include "rafko_gym/services/rafko_backprop_spike_fn_operation.h"

namespace rafko_gym{

void RafkoBackpropSpikeFnOperation::calculate_value(const std::vector<double>& network_input){
  parameter_not_used(network_input);
  RFASSERT(are_dependencies_registered());
  RFASSERT(static_cast<bool>(present_value_dependency));
  RFASSERT(present_value_dependency->is_value_processed());
  double past_value = get_value(1u/*past_index*/);
  set_value(rafko_net::SpikeFunction::get_value(
    network.neuron_array(neuron_index).spike_function(),
    network.weight_table(network.neuron_array(neuron_index).input_weights(0).starts()),
    present_value_dependency->get_value(0u/*past_index*/), past_value
  ));
  set_value_processed();
}

void RafkoBackpropSpikeFnOperation::calculate_derivative(
  std::uint32_t d_w_index,
  const std::vector<double>& network_input, const std::vector<double>& label_data
){
  parameter_not_used(network_input);
  parameter_not_used(label_data);
  RFASSERT(is_value_processed());
  RFASSERT(are_dependencies_registered());
  RFASSERT(static_cast<bool>(present_value_dependency));
  RFASSERT(present_value_dependency->is_processed());
  double past_value = get_value(1u/*past_index*/);
  double past_derivative_value = get_derivative(1u/*past_index*/, d_w_index);
  if(static_cast<std::int32_t>(d_w_index) == network.neuron_array(neuron_index).input_weights(0).starts()){
    set_derivative(d_w_index, rafko_net::SpikeFunction::get_derivative_for_w(
      network.neuron_array(neuron_index).spike_function(),
      network.weight_table(network.neuron_array(neuron_index).input_weights(0).starts()),
      present_value_dependency->get_value(0u/*past_index*/),
      present_value_dependency->get_derivative(0u/*past_index*/, d_w_index),
      past_value, past_derivative_value
    ));
  }else{
    set_derivative(d_w_index, rafko_net::SpikeFunction::get_derivative_not_for_w(
      network.neuron_array(neuron_index).spike_function(),
      network.weight_table(network.neuron_array(neuron_index).input_weights(0).starts()),
      present_value_dependency->get_value(0u/*past_index*/),
      present_value_dependency->get_derivative(0u/*past_index*/, d_w_index),
      past_value, past_derivative_value
    ));
  }
  set_derivative_processed();
}

#if(RAFKO_USES_OPENCL)
std::string RafkoBackpropSpikeFnOperation::value_kernel_operation(
  std::string /*network_input_array*/, std::string /*network_input_array_start*/,
  std::string weight_array, std::string weight_array_start,
  std::string operations_value_array, std::string operations_value_array_start,
  std::string operations_array_size
) const{
  RFASSERT(are_dependencies_registered());
  RFASSERT(static_cast<bool>(present_value_dependency));
  std::string kernel_code = R"(
    double past_value;
    if(0 < available_memory_slots){
      past_value = ==op_value_array==[==op_value_array_start== - ==op_array_size== + ==op_index==];
    }else{
      past_value = 0.0;
    }
    ==op_value_array==[==op_value_array_start== + ==op_index==] = ==spike_kernel==;
  )";
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==spike_kernel=="), rafko_net::SpikeFunction::get_kernel_function_for(
      network.neuron_array(neuron_index).spike_function(),
      weight_array + "[" + weight_array_start + " + " + std::to_string(network.neuron_array(neuron_index).input_weights(0).starts()) + "]",
      "==op_value_array==[==op_value_array_start== + ==value_dep_op_index==]", "past_value"
    )
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==op_value_array=="), operations_value_array
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==op_value_array_start=="), operations_value_array_start
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==op_array_size=="), operations_array_size
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==op_index=="), std::to_string(get_operation_index())
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==value_dep_op_index=="), std::to_string(present_value_dependency->get_operation_index())
  );
  return kernel_code;
}

std::string RafkoBackpropSpikeFnOperation::derivative_kernel_operation(
    std::string /*network_input_array*/, std::string /*network_input_array_start*/,
    std::string /*label_array*/, std::string /*label_array_start*/,
    std::string weight_array, std::string weight_array_start,
    std::string operations_value_array, std::string operations_value_array_start,
    std::string operations_derivative_array, std::string operations_derivative_array_start,
    std::string operations_array_size
) const{
  RFASSERT(are_dependencies_registered());
  RFASSERT(static_cast<bool>(present_value_dependency));
  std::string kernel_code = R"(
    double past_value;
    double past_derivative_value;
    if(0 < available_memory_slots){
      past_value = ==op_value_array==[==op_value_array_start== - ==op_array_size== + ==op_index==];
      past_derivative_value = ==op_derivative_array==[==op_derivative_array_start== - ==op_array_size== + ==op_index==];
    }else{
      past_value = 0.0;
      past_derivative_value = 0.0;
    }
  )";

  kernel_code += R"(
    if(d_w_index == ==this_op_weight_index==){
      ==op_derivative_array==[==op_derivative_array_start== + ==op_index==] = ==spike_w_kernel==;
    }else{
      ==op_derivative_array==[==op_derivative_array_start== + ==op_index==] = ==spike_kernel==;
    }
  )";
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==spike_w_kernel=="), rafko_net::SpikeFunction::get_derivative_kernel_for_w(
      network.neuron_array(neuron_index).spike_function(),
      weight_array + "[" + weight_array_start + " + " + std::to_string(network.neuron_array(neuron_index).input_weights(0).starts()) + "]",
      "==op_value_array==[==op_value_array_start== + ==value_dep_op_index==]",
      "==op_derivative_array==[==op_derivative_array_start== + ==value_dep_op_index==]",
      "past_value", "past_derivative_value"
    )
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==spike_kernel=="), rafko_net::SpikeFunction::get_derivative_kernel_not_for_w(
      network.neuron_array(neuron_index).spike_function(),
      weight_array + "[" + weight_array_start + " + " + std::to_string(network.neuron_array(neuron_index).input_weights(0).starts()) + "]",
      "==op_derivative_array==[==op_derivative_array_start== + ==value_dep_op_index==]",
      "past_derivative_value"
    )
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==this_op_weight_index=="),
    std::to_string( network.neuron_array(neuron_index).input_weights(0).starts() )
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==op_value_array=="), operations_value_array
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==op_value_array_start=="), operations_value_array_start
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==op_array_size=="), operations_array_size
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==op_derivative_array=="), operations_derivative_array
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==op_derivative_array_start=="), operations_derivative_array_start
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==op_index=="), std::to_string(get_operation_index())
  );
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, std::regex("==value_dep_op_index=="), std::to_string(present_value_dependency->get_operation_index())
  );
  return kernel_code;
}
#endif/*(RAFKO_USES_OPENCL)*/

} /* namespace rafko_gym */

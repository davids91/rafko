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
#include "rafko_gym/services/rafko_backprop_neuron_bias_operation.h"

#if(RAFKO_USES_OPENCL)
#include <regex>

#include "rafko_utilities/services/rafko_string_utils.h"
#endif/*(RAFKO_USES_OPENCL)*/

namespace rafko_gym{

void RafkoBackpropNeuronBiasOperation::calculate_value(const std::vector<double>& network_input){
  parameter_not_used(network_input);
  RFASSERT(are_dependencies_registered());
  if(neuron_weight_index < (weights_iterator.cached_size() - 1u)){
    RFASSERT(static_cast<bool>(next_bias_dependency));
    RFASSERT(next_bias_dependency->is_value_processed());
    set_value(rafko_net::InputFunction::collect(
      network.neuron_array(neuron_index).input_function(),
      network.weight_table(weight_index), next_bias_dependency->get_value(0u/*past_index*/)
    ));
  }else{ /* no additional bias values are present as dependencies */
    set_value(network.weight_table(weight_index));
  }
  set_value_processed();
}

void RafkoBackpropNeuronBiasOperation::calculate_derivative(
  std::uint32_t d_w_index, const std::vector<double>& network_input, const std::vector<double>& label_data
){
  parameter_not_used(network_input);
  parameter_not_used(label_data);
  RFASSERT(is_value_processed());
  RFASSERT(are_dependencies_registered());
  if(neuron_weight_index < (weights_iterator.cached_size() - 1u)){
    RFASSERT(static_cast<bool>(next_bias_dependency));
    RFASSERT(next_bias_dependency->is_processed());
    set_derivative(d_w_index, rafko_net::InputFunction::get_derivative(
      network.neuron_array(neuron_index).input_function(),
      network.weight_table(weight_index), ((d_w_index == weight_index)?(1.0):(0.0)),
      next_bias_dependency->get_value(0u/*past_index*/),
      next_bias_dependency->get_derivative(0u/*past_index*/, d_w_index)
    ));
  }else{ /* no additional bias values are present as dependencies */
    set_derivative( d_w_index, ((d_w_index == weight_index)?(1.0):(0.0)) );
  }
  set_derivative_processed();
}

#if(RAFKO_USES_OPENCL)
std::string RafkoBackpropNeuronBiasOperation::value_kernel_operation(
  std::string /*network_input_array*/, std::string /*network_input_array_start*/,
  std::string weight_array, std::string weight_array_start,
  std::string operations_value_array, std::string operations_value_array_start,
  std::string /*operations_array_size*/
) const{
  if(neuron_weight_index < (weights_iterator.cached_size() - 1u)){
    RFASSERT(static_cast<bool>(next_bias_dependency));
    return(
      operations_value_array + "[" + operations_value_array_start + " + " + std::to_string(operation_index) + "] = "
      + rafko_net::InputFunction::get_kernel_function_for(
        network.neuron_array(neuron_index).input_function(),
        weight_array + "[" + weight_array_start + " + " + std::to_string(weight_index) + "]",
        operations_value_array + "["
          + operations_value_array_start + " + "
          + std::to_string(next_bias_dependency->get_operation_index())
        + "]"
      )
    );
  }else{ /* no additional bias values are present as dependencies */
    return (
      operations_value_array + "["
        + operations_value_array_start + " + " + std::to_string(operation_index)
      + "] = "
      + weight_array + "["
        + weight_array_start + " + " + std::to_string(weight_index)
      + "];"
    );
  }
}

std::string RafkoBackpropNeuronBiasOperation::derivative_kernel_operation(
  std::string /*network_input_array*/, std::string /*network_input_array_start*/,
  std::string /*label_array*/, std::string /*label_array_start*/,
  std::string weight_array, std::string weight_array_start,
  std::string operations_value_array, std::string operations_value_array_start,
  std::string operations_derivative_array, std::string operations_derivative_array_start,
  std::string /*operations_array_size*/
) const{
  RFASSERT(are_dependencies_registered());
  if(neuron_weight_index < (weights_iterator.cached_size() - 1u)){ /* There is a next bias value! */
    RFASSERT(static_cast<bool>(next_bias_dependency));
    std::string kernel_code = (
      operations_derivative_array + "["
        + operations_derivative_array_start + " + " + std::to_string(get_operation_index())
      + "] = "
      + rafko_net::InputFunction::derivative_kernel_for(
        network.neuron_array(neuron_index).input_function(),
        weight_array + "["
          + weight_array_start + " + " + std::to_string(network.neuron_array(neuron_index).input_weights(0).starts())
        + "]",
        "((d_w_index == ==this_op_weight_index==)?(1.0):(0.0))",
        "==op_value_array==[==op_value_array_start== + ==value_dep_op_index==]",
        "==op_derivative_array==[==op_derivative_array_start== + ==value_dep_op_index==]"
      )
      + ";"
    );
    kernel_code = rafko_utilities::replace_all_in_string(
      kernel_code, std::regex("==op_value_array=="), operations_value_array
    );
    kernel_code = rafko_utilities::replace_all_in_string(
      kernel_code, std::regex("==op_value_array_start=="), operations_value_array_start
    );
    kernel_code = rafko_utilities::replace_all_in_string(
      kernel_code, std::regex("==op_derivative_array=="), operations_derivative_array
    );
    kernel_code = rafko_utilities::replace_all_in_string(
      kernel_code, std::regex("==op_derivative_array_start=="), operations_derivative_array_start
    );
    kernel_code = rafko_utilities::replace_all_in_string(
      kernel_code, std::regex("==value_dep_op_index=="), std::to_string(next_bias_dependency->get_operation_index())
    );
    return kernel_code;
  }else{ /* no additional bias values are present as dependencies */
    std::string kernel_code = (
      operations_derivative_array + "["
        + operations_derivative_array_start + " + " + std::to_string(get_operation_index())
      + "] = ((d_w_index == ==this_op_weight_index==)?(1.0):(0.0));"
    );
    kernel_code = rafko_utilities::replace_all_in_string(
      kernel_code, std::regex("==this_op_weight_index=="), std::to_string(weight_index)
    );
    return kernel_code;
  }
}
#endif/*(RAFKO_USES_OPENCL)*/

} /* namespace rafko_gym */

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
#include "rafko_gym/services/rafko_backprop_neuron_input_operation.h"

namespace rafko_gym{

RafkoBackpropNeuronInputOperation::RafkoBackpropNeuronInputOperation(
  RafkoBackpropagationData& data, const rafko_net::RafkoNet& network,
  std::uint32_t operation_index, std::uint32_t neuron_index_, std::uint32_t neuron_input_index_
)
: RafkoBackpropagationOperation(data, network, operation_index, ad_operation_neuron_input_d)
, neuron_index(neuron_index_)
, neuron_input_index(neuron_input_index_)
, inputs_iterator(network.neuron_array(neuron_index).input_indices())
, weights_iterator(network.neuron_array(neuron_index).input_weights())
, is_network_input(
  rafko_net::SynapseIterator<rafko_net::InputSynapseInterval>::is_index_input(
    inputs_iterator[neuron_input_index]
  )
)
, input_index_from_neuron_input_index(
  (!is_network_input)?(inputs_iterator[neuron_input_index])
  :(
    rafko_net::SynapseIterator<rafko_net::InputSynapseInterval>::synapse_index_from_input_index(
      inputs_iterator[neuron_input_index]
    )
  )
)
, input_past_index(
  inputs_iterator.reach_past_loops<rafko_net::InputSynapseInterval>(neuron_input_index)
)
, weight_index(weights_iterator[1u + neuron_input_index]) /* spike index preceeds the inputs */
{
}

DependencyRequest RafkoBackpropNeuronInputOperation::upload_dependencies_to_operations(){
  DependencyParameters dependency_parameters;
  if(is_network_input){ /* weighted pair from a Neuron or a Network input */
    RFASSERT(0u == input_past_index);
    dependency_parameters.push_back({
      ad_operation_network_input_d,
      {
        input_index_from_neuron_input_index, /*!Note: Network input dependency contains weight */
        static_cast<std::uint32_t>(weights_iterator[1 + neuron_input_index]),
        neuron_index /* debug information */
      }
    });
  }else{ /* if it's not an input, then it's an internal neuron value */
    dependency_parameters.push_back({ad_operation_neuron_spike_d, {input_index_from_neuron_input_index}});
  }
  if(neuron_input_index < (inputs_iterator.cached_size() - 1u)){ /* this is not the last input */
    /* push in dependency u(x) = every input after this one */
    dependency_parameters.push_back({
      ad_operation_neuron_input_d, {neuron_index, (neuron_input_index + 1u)}
    });
    /*!Note: current operation is to calculate the inputs starting from the current index,
     * but the elements starting from the next input is as a dependency.
     */
  }else{ /* this is the last input, push in the bias dependency */
    dependency_parameters.push_back({
      ad_operation_neuron_bias_d, {neuron_index, (1u + neuron_input_index + 1u)}
    });
  }

  return {{dependency_parameters, [this](std::vector<std::shared_ptr<RafkoBackpropagationOperation>> dependencies){
      RFASSERT(1 <= dependencies.size());
      if(is_network_input)
        network_input_dependency = dependencies[0];
        else neuron_data_dependency = dependencies[0];
      if(neuron_input_index < (inputs_iterator.cached_size() - 1u)){
        RFASSERT(2 == dependencies.size());
        neuron_input_dependency = dependencies[1];
      }else{
        RFASSERT(2 == dependencies.size());
        neuron_bias_dependency = dependencies[1];
      }
      set_registered();
    }
  }};
}

void RafkoBackpropNeuronInputOperation::calculate_value(const std::vector<double>& network_input){
  parameter_not_used(network_input);
  RFASSERT(are_dependencies_registered());
  /* i(w) = w * f(w) ¤ u(w) | f(w) = network_input or input_from_internal_neuron */
  /* calculate f(x) part */
  double weighted_input;
  if(is_network_input){ /* f(x) comes from network input, should not get it from the past */
    RFASSERT(0u == input_past_index);
    RFASSERT(static_cast<bool>(network_input_dependency));
    RFASSERT(network_input_dependency->is_value_processed());
    weighted_input = network_input_dependency->get_value(0u/*past_index*/);
  }else{ /* f(x) comes from Neuron data, may have inputs from the past */
    RFASSERT(static_cast<bool>(neuron_data_dependency));
    RFASSERT( (0u < input_past_index)||(neuron_data_dependency->is_value_processed()) );
    weighted_input = ( neuron_data_dependency->get_value(input_past_index) * network.weight_table(weight_index) );
  }/*if(is_network_input)*/

  /* calculate u(x) part, u(x) is either the inputs starting from the next, or the bias value(s) */
  double next_value = 0.0;
  if(neuron_input_index < (inputs_iterator.cached_size() - 1u)){
    RFASSERT(static_cast<bool>(neuron_input_dependency));
    RFASSERT(neuron_input_dependency->is_value_processed());
    next_value = neuron_input_dependency->get_value(0u/*past_index*/);
  }else{ /* the last input starts to collect bias */
    RFASSERT(static_cast<bool>(neuron_bias_dependency));
    RFASSERT(neuron_bias_dependency->is_value_processed());
    next_value = neuron_bias_dependency->get_value(0u/*past_index*/);
  }
  /* calculate the overall value */
  set_value( rafko_net::InputFunction::collect(
    network.neuron_array(neuron_index).input_function(), weighted_input, next_value
  ) );
  set_value_processed();
}

void RafkoBackpropNeuronInputOperation::calculate_derivative(
  std::uint32_t d_w_index, const std::vector<double>& network_input, const std::vector<double>& label_data
){
  parameter_not_used(network_input);
  parameter_not_used(label_data);
  RFASSERT(is_value_processed());
  RFASSERT(are_dependencies_registered());
  /* i(w) = w * f(w) ¤ u(w) | f(w) = network_input or input_from_internal_neuron */
  /* calculate f(x) part */
  double f_x_derivative;
  if(is_network_input){
    RFASSERT(0u == input_past_index);
    RFASSERT(static_cast<bool>(network_input_dependency));
    RFASSERT(network_input_dependency->is_processed());
    f_x_derivative = network_input_dependency->get_derivative(0u/*past_index*/, d_w_index);
  }else{ /* f(x) comes from Neuron data, may have inputs from the past */
    RFASSERT(static_cast<bool>(neuron_data_dependency));
    RFASSERT( (0u < input_past_index)||(neuron_data_dependency->is_processed()) );
    f_x_derivative = (
      neuron_data_dependency->get_derivative(input_past_index, d_w_index)
      * network.weight_table(weight_index)
    );
    if(weight_index == d_w_index)f_x_derivative += get_value(0u/*past_index*/);
  }/*if(is_network_input)*/

  /* calculate u(x) part, u(x) is either the inputs starting from the next, or the bias value(s) */
  double next_value = 0.0;
  double next_derivative = 0.0;
  if(neuron_input_index < (inputs_iterator.cached_size() - 1u)){
    RFASSERT(static_cast<bool>(neuron_input_dependency));
    RFASSERT(neuron_input_dependency->is_processed());
    next_value = neuron_input_dependency->get_value(0u/*past_index*/);
    next_derivative = neuron_input_dependency->get_derivative(0u/*past_index*/, d_w_index);
  }else{ /* the last input starts to collect bias */
    RFASSERT(static_cast<bool>(neuron_bias_dependency));
    RFASSERT(neuron_bias_dependency->is_processed());
    next_value = neuron_bias_dependency->get_value(0u/*past_index*/);
    next_derivative = neuron_bias_dependency->get_derivative(0u/*past_index*/, d_w_index);
  }
  /* calculate the derivative part */
  set_derivative(d_w_index, rafko_net::InputFunction::get_derivative(
    network.neuron_array(neuron_index).input_function(),
    get_value(0u/*past_index*/), f_x_derivative, next_value, next_derivative
  ));
  set_derivative_processed();
}

#if(RAFKO_USES_OPENCL)
std::string RafkoBackpropNeuronInputOperation::value_kernel_function(
  std::string network_input_array, std::string network_input_array_start,
  std::string weight_array, std::string weight_array_start,
  std::string operations_value_array, std::string operations_value_array_start,
  std::string operations_array_size
) const{
  std::string operations = "double weighted_input = 0;\n";

  /* Calculate the weighted input(f(x)) */
  if(is_network_input){ /* f(x) comes from network input, should not get it from the past */
    RFASSERT(0u == input_past_index);
    RFASSERT(static_cast<bool>(network_input_dependency));
    operations += "weighted_input = " + operations_value_array + "["
      + operations_value_array_start + " + "
      + std::to_string(network_input_dependency->get_operation_index())
    + "];\n";
  }else{ /* f(x) comes from Neuron data, may have inputs from the past */
    RFASSERT(static_cast<bool>(neuron_data_dependency));
    RFASSERT( (0u < input_past_index)||(neuron_data_dependency->is_value_processed()) );
    operations += R"(
      if(==past_index== <= available_memory_slots){
        weighted_input = ==op_value_array==[
          ==op_value_array_start== + ==op_index==
          - (==op_value_array_size== * ==past_index==)
        ];
      }else{
        weighted_input = 0.0;
      }
    )";
  }/*if(is_network_input)*/

  /* calculate the next value (u(x)) */
  if(neuron_input_index < (inputs_iterator.cached_size() - 1u)){
    RFASSERT(static_cast<bool>(neuron_input_dependency));
    operations += "weighted_input = " + operations_value_array + "["
      + operations_value_array_start + " + "
      + std::to_string(neuron_input_dependency->get_operation_index())
    + "];\n";
  }else{ /* the last input starts to collect bias */
    RFASSERT(static_cast<bool>(neuron_bias_dependency));
    operations += "weighted_input = " + operations_value_array + "["
      + operations_value_array_start + " + "
      + std::to_string(neuron_bias_dependency->get_operation_index())
    + "];\n";
  }

  /* add the input function */
  operations += rafko_net::InputFunction::get_kernel_function_for(
    network.neuron_array(neuron_index).input_function(), "weighted_input", "next_value"
  ) + ";\n";
  return operations;
}

std::string RafkoBackpropNeuronInputOperation::derivative_kernel_function(
  std::string network_input_array, std::string network_input_array_start,
  std::string weight_array, std::string weight_array_start,
  std::string operations_value_array, std::string operations_value_array_start,
  std::string operations_array_size
) const{
  RFASSERT(are_dependencies_registered());
  std::string kernel_code = R"(
    double f_x_value;
    double f_x_derivative;
    double u_x_value = ==op_value_array==[==op_value_array_start== + ==u_x_op_index==];
    double u_x_derivative = ==op_derivative_array==[==op_derivative_array_start== + ==u_x_op_index==];
    if(==past_index== < available_memory_slots){
      f_x_value = ==op_value_array==[
        ==op_value_array_start== - (==op_array_size== * ==past_index==) + ==dep_op_index==
      ];
      f_x_derivative = ==op_derivative_array==[
        ==op_derivative_array_start== - (==op_array_size== * ==past_index==) + ==dep_op_index==
      ];
    }else{
      f_x_value = 0.0;
      f_x_derivative = 0.0;
    }
    ==op_derivative_array==[==op_derivative_array_start== + ==op_index==] = ==input_kernel==;
  )";
  if(is_network_input){
    RFASSERT(0u == input_past_index);
    RFASSERT(static_cast<bool>(network_input_dependency));
    kernel_code = rafko_utilities::replace_all_in_string(
      kernel_code, "==dep_op_index==", std::to_string(network_input_dependency->get_operation_index())
    );
  }else{
    RFASSERT(static_cast<bool>(neuron_data_dependency));
    RFASSERT( (0u < input_past_index)||(neuron_data_dependency->is_processed()) );
    kernel_code = rafko_utilities::replace_all_in_string(
      kernel_code, "==dep_op_index==", std::to_string(neuron_data_dependency->get_operation_index())
    );
  }

  if(neuron_input_index < (inputs_iterator.cached_size() - 1u)){
    RFASSERT(static_cast<bool>(neuron_input_dependency));
    RFASSERT(neuron_input_dependency->is_processed());
    kernel_code = rafko_utilities::replace_all_in_string(
      kernel_code, "==u_x_op_index==", std::to_string(neuron_input_dependency->get_operation_index())
    );
  }else{ /* the last input starts to collect bias */
    RFASSERT(static_cast<bool>(neuron_bias_dependency));
    RFASSERT(neuron_bias_dependency->is_processed());
    kernel_code = rafko_utilities::replace_all_in_string(
      kernel_code, "==u_x_op_index==", std::to_string(neuron_bias_dependency->get_operation_index())
    );
  }
  kernel_code = rafko_utilities::replace_all_in_string(
    kernel_code, "==input_kernel==",
    rafko_net::InputFunction::get_derivative(
      network.neuron_array(neuron_index).input_function(),
      "f_x_value","u_x_derivative","u_x_value","u_x_derivative"
    )
  );
  kernel_code = rafko_utilities::replace_all_in_string(kernel_code, "==op_index==", std::to_string(get_operation_index));
  kernel_code = rafko_utilities::replace_all_in_string(kernel_code, "==op_value_array==", operations_value_array);
  kernel_code = rafko_utilities::replace_all_in_string(kernel_code, "==op_value_array_start==", operations_value_array_start);
  kernel_code = rafko_utilities::replace_all_in_string(kernel_code, "==op_array_size==", operations_array_size);
  kernel_code = rafko_utilities::replace_all_in_string(kernel_code, "==op_derivative_array==", operations_derivative_array);
  kernel_code = rafko_utilities::replace_all_in_string(kernel_code, "==op_derivative_array_start==", operations_derivative_array_start);
  return kernel_code;
}
#endif/*(RAFKO_USES_OPENCL)*/


} /* namespace rafko_gym */

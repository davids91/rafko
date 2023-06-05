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
#include "rafko_gym/services/rafko_backprop_neuron_input_operation.hpp"

#include "rafko_utilities/services/rafko_string_utils.hpp"

namespace rafko_gym{

RafkoBackpropNeuronInputOperation::RafkoBackpropNeuronInputOperation(
  RafkoBackpropagationData& data, const rafko_net::RafkoNet& network,
  std::uint32_t operation_index, std::uint32_t neuron_index, std::uint32_t m_neuronInputIndex
)
: RafkoBackpropagationOperation(data, network, operation_index, ad_operation_neuron_input_d)
, m_neuronIndex(neuron_index)
, m_neuronInputIndex(m_neuronInputIndex)
, m_inputsIterator(m_network.neuron_array(m_neuronIndex).input_indices())
, m_weightsIterator(m_network.neuron_array(m_neuronIndex).input_weights())
, m_isNetworkInput(
  rafko_net::SynapseIterator<rafko_net::InputSynapseInterval>::is_index_input(
    m_inputsIterator[m_neuronInputIndex]
  )
)
, m_inputPastIndex(
  m_inputsIterator.reach_past_loops<rafko_net::InputSynapseInterval>(m_neuronInputIndex)
)
, m_weightIndex(m_weightsIterator[1u + m_neuronInputIndex]) /* spike index preceeds the inputs(so +1 offset is needed) */
{
}

RafkoBackpropagationOperation::DependencyRequest RafkoBackpropNeuronInputOperation::upload_dependencies_to_operations(){
  RafkoBackpropagationOperation::DependencyParameters dependency_parameters;
  if(m_isNetworkInput){ /* weighted pair from a Neuron or a Network input */
    RFASSERT(0u == m_inputPastIndex);
    dependency_parameters.push_back({
      ad_operation_network_input_d,
      {
        rafko_net::SynapseIterator<rafko_net::InputSynapseInterval>::array_index_from_external_index(
          m_inputsIterator[m_neuronInputIndex]
        ), /* index inside the input array for the network */
        static_cast<std::uint32_t>(m_weightsIterator[1 + m_neuronInputIndex]), /* weight index to be used with the input */
        m_neuronIndex /* debug information */
      }
    });
  }else{ /* if it's not an input, then it's an internal neuron value */
    dependency_parameters.push_back({
      ad_operation_neuron_spike_d, {static_cast<std::uint32_t>(m_inputsIterator[m_neuronInputIndex])}
    });
  }
  if(m_neuronInputIndex < (m_inputsIterator.cached_size() - 1u)){ /* this is not the last input */
    /* push in dependency u(x) = every input after this one */
    dependency_parameters.push_back({
      ad_operation_neuron_input_d, {m_neuronIndex, (m_neuronInputIndex + 1u)}
    });
    /*!Note: current operation is to calculate the inputs starting from the current index,
     * but the elements starting from the next input is as a dependency.
     */
  }else{ /* this is the last input, push in the bias dependency */
    dependency_parameters.push_back({
      ad_operation_neuron_bias_d, {m_neuronIndex, (1u + m_neuronInputIndex + 1u)}
    });
  }

  return {{dependency_parameters, [this](std::vector<std::shared_ptr<RafkoBackpropagationOperation>> dependencies){
      if(m_isNetworkInput){
        RFASSERT(1 <= dependencies.size());
        RFASSERT(static_cast<bool>(dependencies[0]));
        RFASSERT_LOG(
          "Neuron input operation[{}]: Registering dependency: operation[{}(?)] as network input based on [{}]",
          get_operation_index(), dependencies[0]->get_operation_index(), m_inputsIterator[m_neuronInputIndex]
        );
        m_networkInputDependency = dependencies[0];
      }else{
        RFASSERT(1 <= dependencies.size());
        RFASSERT(static_cast<bool>(dependencies[0]));
        RFASSERT_LOG(
          "Neuron input operation[{}]: Registering dependency: operation[{}(?)] as neuron data input based on [{}]",
          get_operation_index(), dependencies[0]->get_operation_index(), m_inputsIterator[m_neuronInputIndex]
        );
        m_neuronDataDependency = dependencies[0];
      }

      if(m_neuronInputIndex < (m_inputsIterator.cached_size() - 1u)){
        RFASSERT(2 == dependencies.size());
        RFASSERT(static_cast<bool>(dependencies[1]));
        RFASSERT_LOG(
          "Neuron input operation[{}]: Registering dependency: operation[{}(?)] as other neuron input",
          get_operation_index(), dependencies[1]->get_operation_index()
        );
        m_neuronInputDependency = dependencies[1];
      }else{
        RFASSERT(2 == dependencies.size());
        RFASSERT(static_cast<bool>(dependencies[1]));
        RFASSERT_LOG(
          "Neuron input operation[{}]: Registering dependency: operation[{}(?)] as neuron bias",
          get_operation_index(), dependencies[1]->get_operation_index()
        );
        m_neuronBiasDependency = dependencies[1];
      }
      set_registered();
    }
  }};
}

std::vector<std::shared_ptr<RafkoBackpropagationOperation>> RafkoBackpropNeuronInputOperation::get_own_dependencies(){
  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> dependencies;
  if(m_networkInputDependency)
    dependencies.push_back(m_networkInputDependency);
  if(m_neuronDataDependency && (0 == m_inputPastIndex))
    dependencies.push_back(m_neuronDataDependency);
  if(m_neuronInputDependency)
    dependencies.push_back(m_neuronInputDependency);
  if(m_neuronBiasDependency)
    dependencies.push_back(m_neuronBiasDependency);
  return dependencies;
}


std::vector<std::shared_ptr<RafkoBackpropagationOperation>> RafkoBackpropNeuronInputOperation::get_own_dependencies_past_included(){
  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> dependencies;
  if(m_networkInputDependency)
    dependencies.push_back(m_networkInputDependency);
  if(m_neuronDataDependency)
    dependencies.push_back(m_neuronDataDependency);
  if(m_neuronInputDependency)
    dependencies.push_back(m_neuronInputDependency);
  if(m_neuronBiasDependency)
    dependencies.push_back(m_neuronBiasDependency);
  RFASSERT(2u == dependencies.size() || !are_dependencies_registered());
  return dependencies;
}

void RafkoBackpropNeuronInputOperation::calculate_value(const std::vector<double>& /*network_input*/){
  RFASSERT(are_dependencies_registered());
  /* i(w) = w * f(w) ¤ u(w) | f(w) = network_input or input_from_internal_neuron */
  /* calculate f(x) part */
  double weighted_input;
  if(m_isNetworkInput){ /* f(x) comes from network input, should not get it from the past */
    RFASSERT(0u == m_inputPastIndex);
    RFASSERT(static_cast<bool>(m_networkInputDependency));
    RFASSERT(m_networkInputDependency->is_value_processed());
    weighted_input = m_networkInputDependency->get_value(0u/*past_index*/);
    RFASSERT_LOG(
      "operation[{}]: Neuron[{}] Input[{}] f_x = {}(op[{}])",
      get_operation_index(), m_neuronIndex, m_neuronInputIndex,
      weighted_input, m_networkInputDependency->get_operation_index()
    );
  }else{ /* f(x) comes from Neuron data, may have inputs from the past */
    RFASSERT(static_cast<bool>(m_neuronDataDependency));
    RFASSERT( (0u < m_inputPastIndex)||(m_neuronDataDependency->is_value_processed()) );
    weighted_input = ( m_neuronDataDependency->get_value(m_inputPastIndex) * m_network.weight_table(m_weightIndex) );
    RFASSERT_LOG(
      "operation[{}]: Neuron[{}] Input[{}] f_x = op[{}](past:{} = {}) * weight[{}]({}) = {}",
      get_operation_index(), m_neuronIndex, m_neuronInputIndex,
      m_neuronDataDependency->get_operation_index(), m_inputPastIndex, m_neuronDataDependency->get_value(m_inputPastIndex),
      m_weightIndex, m_network.weight_table(m_weightIndex), weighted_input
    );
  }/*if(is_network_input)*/

  /* calculate u(x) part, u(x) is either the inputs starting from the next, or the bias value(s) */
  double next_value = 0.0;
  if(m_neuronInputIndex < (m_inputsIterator.cached_size() - 1u)){
    RFASSERT(static_cast<bool>(m_neuronInputDependency));
    RFASSERT(m_neuronInputDependency->is_value_processed());
    next_value = m_neuronInputDependency->get_value(0u/*past_index*/);
    RFASSERT_LOG(
      "operation[{}]: Neuron[{}] Input[{}] u_x = {}(op[{}])",
      get_operation_index(), m_neuronIndex, m_neuronInputIndex,
      next_value, m_neuronInputDependency->get_operation_index()
    );
  }else{ /* the last input starts to collect bias */
    RFASSERT(static_cast<bool>(m_neuronBiasDependency));
    RFASSERT(m_neuronBiasDependency->is_value_processed());
    next_value = m_neuronBiasDependency->get_value(0u/*past_index*/);
    RFASSERT_LOG(
      "operation[{}]: Neuron[{}] Input[{}] u_x = b{}(op[{}])", get_operation_index(), m_neuronIndex, m_neuronInputIndex,
      next_value, m_neuronBiasDependency->get_operation_index()
    );
  }
  /* calculate the overall value */
  set_value( rafko_net::InputFunction::collect(
    get_input_function(), weighted_input, next_value
  ) );
  RFASSERT_LOG(
    "operation[{}]: Neuron[{}] Input[{}] = {} (collected with {})",
    get_operation_index(), m_neuronIndex, m_neuronInputIndex,
    get_value(0u/*past_index*/), Input_functions_Name(get_input_function())
  );
  set_value_processed();
}

void RafkoBackpropNeuronInputOperation::calculate_derivative(
  std::uint32_t d_w_index, const std::vector<double>& /*network_input*/, const std::vector<double>& /*label_data*/
){
  RFASSERT(is_value_processed());
  RFASSERT(are_dependencies_registered());
  /* i(w) = w * f(w) ¤ u(w) | f(w) = network_input or input_from_internal_neuron */
  /* calculate f(x) part */
  double f_x_value;
  double f_x_derivative;
  if(m_isNetworkInput){
    RFASSERT(0u == m_inputPastIndex);
    RFASSERT(static_cast<bool>(m_networkInputDependency));
    RFASSERT(m_networkInputDependency->is_processed());
    f_x_value = m_networkInputDependency->get_value(0u/*past_index*/);
    f_x_derivative = m_networkInputDependency->get_derivative(0u/*past_index*/, d_w_index);
    RFASSERT_LOG(
      "derivative_operation[{}](w[{}]): Neuron[{}] Input[{}]_d f_x = {}; f_x_d = {}(d_op[{}])",
      get_operation_index(), d_w_index, m_neuronIndex, m_neuronInputIndex,
      f_x_value, f_x_derivative, m_networkInputDependency->get_operation_index()
    );
  }else{ /* f(x) comes from Neuron data, may have inputs from the past */
    RFASSERT(static_cast<bool>(m_neuronDataDependency));
    RFASSERT( (0u < m_inputPastIndex)||(m_neuronDataDependency->is_processed()) );
    f_x_value = m_neuronDataDependency->get_value(m_inputPastIndex);
    f_x_derivative = (
      m_neuronDataDependency->get_derivative(m_inputPastIndex, d_w_index)
      * m_network.weight_table(m_weightIndex)
    );
    if(m_weightIndex == d_w_index){
      f_x_derivative += f_x_value;
      RFASSERT_LOG(
        "derivative_operation[{}](w[{}]): Neuron[{}] Input[{}]_d f_x = {}; f_x_d = {} = ({}(d_op[{}]) * {}(weight[{}])) + f_x",
        get_operation_index(), d_w_index, m_neuronIndex, m_neuronInputIndex,
        f_x_value, f_x_derivative, m_neuronDataDependency->get_derivative(m_inputPastIndex, d_w_index),
        m_neuronDataDependency->get_operation_index(),
        m_network.weight_table(m_weightIndex), m_weightIndex
      );
    }else{
      RFASSERT_LOG(
        "derivative_operation[{}](w[{}]): Neuron[{}] Input[{}]_d f_x = {}; f_x_d = {} = ({}(d_op[{}]) * {}(weight[{}]))",
        get_operation_index(), d_w_index, m_neuronIndex, m_neuronInputIndex,
        f_x_value, f_x_derivative, m_neuronDataDependency->get_derivative(m_inputPastIndex, d_w_index),
        m_neuronDataDependency->get_operation_index(),
        m_network.weight_table(m_weightIndex), m_weightIndex
      );
    }

  }/*if(is_network_input)*/

  /* calculate u(x) part, u(x) is either the inputs starting from the next, or the bias value(s) */
  double u_x_value = 0.0;
  double u_x_derivative = 0.0;
  if(m_neuronInputIndex < (m_inputsIterator.cached_size() - 1u)){
    RFASSERT(static_cast<bool>(m_neuronInputDependency));
    RFASSERT(m_neuronInputDependency->is_processed());
    u_x_value = m_neuronInputDependency->get_value(0u/*past_index*/);
    u_x_derivative = m_neuronInputDependency->get_derivative(0u/*past_index*/, d_w_index);
    RFASSERT_LOG(
      "derivative_operation[{}](w[{}]): Neuron[{}] Input[{}]_d u_x = {}(op[{}]); u_x_d = {}(d_op[{}])",
      get_operation_index(), d_w_index, m_neuronIndex, m_neuronInputIndex,
      u_x_value, m_neuronInputDependency->get_operation_index(),
      u_x_derivative, m_neuronInputDependency->get_operation_index()
    );
  }else{ /* the last input starts to collect bias */
    RFASSERT(static_cast<bool>(m_neuronBiasDependency));
    RFASSERT(m_neuronBiasDependency->is_processed());
    u_x_value = m_neuronBiasDependency->get_value(0u/*past_index*/);
    u_x_derivative = m_neuronBiasDependency->get_derivative(0u/*past_index*/, d_w_index);
    RFASSERT_LOG(
      "derivative_operation[{}](w[{}]): Neuron[{}] Input[{}]_d u_x = {}(op[{}]); u_x_d = {}(d_op[{}])(bias)",
      get_operation_index(), d_w_index, m_neuronIndex, m_neuronInputIndex,
      u_x_value, m_neuronBiasDependency->get_operation_index(),
      u_x_derivative, m_neuronBiasDependency->get_operation_index()
    );
  }
  /* calculate the derivative part */
  set_derivative(d_w_index, rafko_net::InputFunction::get_derivative(
    get_input_function(), f_x_value, f_x_derivative, u_x_value, u_x_derivative
  ));
  RFASSERT_LOG(
    "derivative operation[{}](w[{}]): Neuron[{}] Input[{}]_d = {} (calculated with {})",
    get_operation_index(), d_w_index, m_neuronIndex, m_neuronInputIndex,
    f_x_derivative, Input_functions_Name(get_input_function())
  );
  set_derivative_processed();
}

#if(RAFKO_USES_OPENCL)
std::string RafkoBackpropNeuronInputOperation::local_declaration_operation() const{
  return R"(
    /* Neuron input operation locals */
    double f_x_value;
    double u_x_value;
    double f_x_derivative;
    double u_x_derivative;
  )";
}

std::string RafkoBackpropNeuronInputOperation::generic_value_kernel_operation(
  std::string weight_array, std::string operations_value_array, std::string operations_array_size,
  std::string behavior_index
){
  std::string operations = R"(
    /* calculate the next value (u(x)) */
    u_x_value = ==op_value_array==[==u_x_op_index==];    
    /* Calculate the weighted input(f(x)) */
    if(==past_index== <= available_memory_slots){
      if(==weight_descriptor== != 0xFFFFFFFFu){ /* this value would mean that the weight is not used */
        f_x_value = (
          ==op_value_array==[(long int)(==f_x_op_index==) - (long int)(==op_value_array_size== * ==past_index==) ] * ==weight_array==[==weight_descriptor==]
        );        
      }else{ /* When weight is not used, it is supposed that the input of the operation is a network input ( already contains weight multiplication ) */
        f_x_value = ==op_value_array==[==f_x_op_index==]; /* It is also supposed that past index is 0 in this case */
      }
    }else{
      f_x_value = 0.0;
    }
  )";

  /* add the input function */
  operations += rafko_net::InputFunction::get_all_kernel_value_functions(behavior_index, "==op_value_array==[==op_index==]", "f_x_value", "u_x_value");

  /* Replacing the tokens with actual kernel string values */
  operations = rafko_utilities::replace_all_in_string(operations, std::regex("==weight_array=="), weight_array);
  operations = rafko_utilities::replace_all_in_string(operations, std::regex("==op_value_array=="), operations_value_array);
  operations = rafko_utilities::replace_all_in_string(operations, std::regex("==op_value_array_size=="), operations_array_size);
  return operations;
}

std::string RafkoBackpropNeuronInputOperation::value_kernel_operation(
  std::string /*network_input_array*/, std::string weight_array,
  std::string operations_value_array, std::string operations_array_size
) const{
  std::string kernel_source = "";

  /* Calculate the weighted input(f(x)) */
  if(m_isNetworkInput){ /* f(x) comes from network input, should not get it from the past */
    RFASSERT(0u == m_inputPastIndex);
    RFASSERT(static_cast<bool>(m_networkInputDependency));
    kernel_source += "f_x_value = " + operations_value_array + "[==f_x_op_index==];\n";
  }else{ /* f(x) comes from Neuron data, may have inputs from the past */
    /*!Note: Past values are supposed to be mapped just before the current array, so
     * the negative index should contain the previous run. It the responsibility of the caller
     * to make sure there is no out pf bounds error with these index values.
     */
    RFASSERT(static_cast<bool>(m_neuronDataDependency));
    kernel_source += R"(
      if(==past_index== <= available_memory_slots){
        f_x_value = (
          ==op_value_array==[(long int)(==f_x_op_index==) - (long int)(==op_value_array_size== * ==past_index==) ] 
          * ==weight_array==[==weight_descriptor==]
        );
      }else{
        f_x_value = 0.0;
      }
    )";
  }/*if(is_network_input)*/

  /* calculate the next value (u(x)) */
  kernel_source += "u_x_value = ==op_value_array==[==u_x_op_index==];\n";

  /* add the input function */
  kernel_source += (
    "==op_value_array==[==op_index==] = "
    + rafko_net::InputFunction::get_kernel_function_for(get_input_function(), "f_x_value", "u_x_value") + ";\n"
  );

  /* Replacing the tokens with actual kernel string values */
  kernel_source = rafko_utilities::replace_all_in_string(kernel_source, std::regex("==weight_array=="), weight_array);
  kernel_source = rafko_utilities::replace_all_in_string(kernel_source, std::regex("==op_value_array=="), operations_value_array);
  kernel_source = rafko_utilities::replace_all_in_string(kernel_source, std::regex("==op_value_array_size=="), operations_array_size);
  kernel_source = rafko_utilities::replace_all_in_string(kernel_source, std::regex("==past_index=="), std::to_string(m_inputPastIndex));
  kernel_source = rafko_utilities::replace_all_in_string(kernel_source, std::regex("==op_index=="), std::to_string(get_operation_index()));
  kernel_source = rafko_utilities::replace_all_in_string(kernel_source, std::regex("==weight_descriptor=="), std::to_string(get_weight_descriptor()));
  if(m_isNetworkInput){
    RFASSERT(0u == m_inputPastIndex);
    RFASSERT(static_cast<bool>(m_networkInputDependency));
    kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==f_x_op_index=="), std::to_string(m_networkInputDependency->get_operation_index())
    );
  }else{
    RFASSERT(static_cast<bool>(m_neuronDataDependency));
    kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==f_x_op_index=="), std::to_string(m_neuronDataDependency->get_operation_index())
    );
  }
  if(m_neuronInputIndex < (m_inputsIterator.cached_size() - 1u)){
    RFASSERT(static_cast<bool>(m_neuronInputDependency));
    kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==u_x_op_index=="), std::to_string(m_neuronInputDependency->get_operation_index())
    );
  }else{ /* the last input starts to collect bias */
    RFASSERT(static_cast<bool>(m_neuronBiasDependency));
    kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==u_x_op_index=="), std::to_string(m_neuronBiasDependency->get_operation_index())
    );
  }

  return kernel_source;
}

std::string RafkoBackpropNeuronInputOperation::generic_derivative_kernel_operation(
  std::string weight_array, std::string operations_value_array, std::string operations_derivative_array,
  std::string operations_array_size, std::string behavior_index
){
  std::string kernel_source = R"(
    u_x_value = ==op_value_array==[==u_x_op_index==];
    u_x_derivative = ==op_derivative_array==[==u_x_op_index==];
    if(==past_index== <= available_memory_slots){
      f_x_value = ==op_value_array==[(long int)(==f_x_op_index==) - (long int)(==op_array_size== * ==past_index==)];
      if(==weight_descriptor== != 0xFFFFFFFFu){
        f_x_derivative = ==f_x_dependency==;
      }else{
        f_x_derivative = ==f_x_dependency_from_network==;
      }
      if(==weight_descriptor== == d_w_index){
        f_x_derivative += f_x_value;
      }
    }else{
      f_x_value = 0.0;
      f_x_derivative = 0.0;
    }

    ==input_kernel==
  )";

  /* finish f_x_dependency */
  kernel_source = rafko_utilities::replace_all_in_string(
    kernel_source, std::regex("==f_x_dependency_from_network=="),
      "==op_derivative_array==[(long int)(==f_x_op_index==) - (long int)(==op_array_size== * ==past_index==)]"
  );
  kernel_source = rafko_utilities::replace_all_in_string(
    kernel_source, std::regex("==f_x_dependency=="),
      "==op_derivative_array==[(long int)(==f_x_op_index==) - (long int)(==op_array_size== * ==past_index==)] " 
      + std::string(" * ==weight_array==[==weight_descriptor==]")
  );

  /* finish u_x_dependency */
  kernel_source = rafko_utilities::replace_all_in_string(
    kernel_source, std::regex("==input_kernel=="),
    rafko_net::InputFunction::get_all_kernel_derivative_functions(
      behavior_index, "==op_derivative_array==[==op_index==]", 
      "f_x_value", "f_x_derivative", "u_x_value", "u_x_derivative"
    )
  );
  kernel_source = rafko_utilities::replace_all_in_string(kernel_source, std::regex("==weight_array=="), weight_array);
  kernel_source = rafko_utilities::replace_all_in_string(kernel_source, std::regex("==op_value_array=="), operations_value_array);
  kernel_source = rafko_utilities::replace_all_in_string(kernel_source, std::regex("==op_array_size=="), operations_array_size);
  kernel_source = rafko_utilities::replace_all_in_string(kernel_source, std::regex("==op_derivative_array=="), operations_derivative_array);
  return kernel_source;
}
#endif/*(RAFKO_USES_OPENCL)*/


} /* namespace rafko_gym */

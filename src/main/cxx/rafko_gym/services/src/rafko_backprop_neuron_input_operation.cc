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

#include "rafko_mainframe/services/rafko_assertion_logger.hpp"
#include "rafko_utilities/services/rafko_string_utils.hpp"
#include <cstdint>
#include <optional>

namespace {
const auto &s_isNeuronInputFromNetworkInput =
    rafko_net::SynapseIterator<rafko_net::InputSynapseInterval>::is_index_input;
const auto &s_convertToArrayIndex = rafko_net::SynapseIterator<
    rafko_net::InputSynapseInterval>::array_index_from_external_index;
} // namespace

namespace rafko_gym {

RafkoBackpropNeuronInputOperation::RafkoBackpropNeuronInputOperation(
    RafkoBackpropagationData &data, const rafko_net::RafkoNet &network,
    std::uint32_t operation_index, std::uint32_t neuron_index,
    std::uint32_t neuron_input_index)
    : RafkoBackpropagationOperation(data, network, operation_index,
                                    ad_operation_neuron_input_d),
      m_neuronIndex(neuron_index), m_neuronInputIndex(neuron_input_index),
      m_inputsIterator(m_network.neuron_array(m_neuronIndex).input_indices()),
      m_weightsIterator(m_network.neuron_array(m_neuronIndex).input_weights()),
      m_isNextDepBias(
          (((m_neuronInputIndex + 1u) < m_inputsIterator.cached_size()) ||
           ((m_neuronInputIndex + 1u) < (m_weightsIterator.cached_size() - 1u)))
              ? std::optional<bool>((m_neuronInputIndex + 1u) >=
                                     m_inputsIterator.cached_size())
              : std::nullopt),
      m_inputPastIndex(
          (m_neuronInputIndex < m_inputsIterator.cached_size() &&
           s_isNeuronInputFromNetworkInput(
               m_inputsIterator[m_neuronInputIndex]))
              ? std::nullopt
              : std::optional<std::uint32_t>(
                    m_inputsIterator
                        .reach_past_loops<rafko_net::InputSynapseInterval>(
                            m_neuronInputIndex))),
      m_inputIndex(
          m_inputPastIndex.has_value()
              ? m_inputsIterator[m_neuronInputIndex]
              : s_convertToArrayIndex(m_inputsIterator[m_neuronInputIndex])),
      m_weightIndex(
          m_weightsIterator[1u + m_neuronInputIndex]) /* spike index preceeds
                                                         the inputs(so +1 offset
                                                         is needed) */
{}

RafkoBackpropagationOperation::DependencyRequest
RafkoBackpropNeuronInputOperation::request_dependencies() {
  RafkoBackpropagationOperation::DependencyParameters dependency_parameters;
  /* In case a network input, the operation doesn't require a dependency */
  if (m_inputPastIndex.has_value()) {
    /*!Note: Internal Neuron values might have past values. A
            neuron output value is acquired from a spike_fn operation! */
    dependency_parameters.push_back(
        {ad_operation_neuron_spike_d, {m_inputIndex}});
  }
  /* See if next input is a bias or a neuron input dependency */
  if (m_isNextDepBias.has_value() && !m_isNextDepBias.value()) {
    /* push in dependency u(x), which represents 'every input after this one
     * inside this neuron */
    dependency_parameters.push_back(
        {ad_operation_neuron_input_d,
         {m_neuronIndex, (m_neuronInputIndex + 1u)}});
    /*!Note: current operation is to calculate the inputs starting from the
     * current index, but the elements starting from the next input is a
     * dependency.
     */
  } else if (m_isNextDepBias.has_value()) { /* this is the last input, push in
                                               the bias dependency */
    dependency_parameters.push_back(
        {ad_operation_neuron_bias_d,
         {m_neuronIndex, (1u + m_neuronInputIndex + 1u)}});
  }

  return {
      {dependency_parameters,
       [this](std::vector<std::shared_ptr<RafkoBackpropagationOperation>>
                  dependencies) {
         std::uint32_t f_x_dependecy_count = 0u;
         if (m_inputPastIndex.has_value()) {
           f_x_dependecy_count = 1u;
           RFASSERT(1 <= dependencies.size());
           RFASSERT(static_cast<bool>(dependencies[0]));
           RFASSERT_LOG("Neuron input operation[{}]: Registering dependency: "
                        "operation[{}(?)] as neuron data input based on [{}]",
                        get_operation_index(),
                        dependencies[0]->get_operation_index(), m_inputIndex);
           m_neuronDataDependency = dependencies[0];
         }

         /* In case there's an f_x dependecy, the other dependency need be
          * after it */
         RFASSERT_LOG("Neuron input operation[{}]: f_x dependecy count: {}/{}",
                      get_operation_index(), f_x_dependecy_count,
                      dependencies.size());
         if (m_isNextDepBias.has_value()) {
           RFASSERT(f_x_dependecy_count < dependencies.size());
           RFASSERT(static_cast<bool>(dependencies[f_x_dependecy_count]));
           m_nextDependency = dependencies[f_x_dependecy_count];
           RFASSERT_LOG(
               "Neuron input operation[{}]: Registering next dependency: "
               "operation[{}(?)] as {}",
               get_operation_index(),
               dependencies[f_x_dependecy_count]->get_operation_index(),
               *m_isNextDepBias ? "neuron bias" : "other neuron input");
         }
         set_registered();
       }}};
}

std::vector<std::shared_ptr<RafkoBackpropagationOperation>>
RafkoBackpropNeuronInputOperation::get_own_dependencies() {
  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> dependencies;
  if (m_neuronDataDependency && (0 == m_inputPastIndex))
    dependencies.push_back(m_neuronDataDependency);
  if (m_isNextDepBias.has_value()) {
    RFASSERT(static_cast<bool>(m_nextDependency));
    dependencies.push_back(m_nextDependency);
  }

  return dependencies;
}

std::vector<std::shared_ptr<RafkoBackpropagationOperation>>
RafkoBackpropNeuronInputOperation::get_own_dependencies_past_included() {
  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> dependencies;
  if (m_neuronDataDependency)
    dependencies.push_back(m_neuronDataDependency);
  if (m_isNextDepBias.has_value()) {
    RFASSERT(static_cast<bool>(m_nextDependency));
    dependencies.push_back(m_nextDependency);
  }
  return dependencies;
}

void RafkoBackpropNeuronInputOperation::calculate_value(
    const std::vector<double> &network_input) {
  /* i(w) = w * f(w) ¤ u(w) | f(w) = network_input or internal_neuron_input */
  RFASSERT(are_dependencies_registered());

  /* calculate f(x) part */
  double weighted_input;
  if (!m_inputPastIndex.has_value()) { /* f(x) comes from network input */
    weighted_input =
        network_input[m_inputIndex] * m_network.weight_table(m_weightIndex);
    RFASSERT_LOG("operation[{}]: Neuron[{}] Input[{}] f_x = input[{}]({}) * "
                 "weight[{}]({}) = {}",
                 get_operation_index(), m_neuronIndex, m_neuronInputIndex,
                 m_inputIndex, network_input[m_inputIndex], m_weightIndex,
                 m_network.weight_table(m_weightIndex), weighted_input);
  } else { /* f(x) comes from Neuron data, may have inputs from the past */
    RFASSERT(static_cast<bool>(m_neuronDataDependency));
    RFASSERT((0u < *m_inputPastIndex) ||
             (m_neuronDataDependency->is_value_processed()));
    weighted_input = (m_neuronDataDependency->get_value(*m_inputPastIndex) *
                      m_network.weight_table(m_weightIndex));
    RFASSERT_LOG(
        "operation[{}]: Neuron[{}] Input[{}] f_x = neuron[] spike, "
        "op[{}](past:{} = {}) * weight[{}]({}) = {}",
        get_operation_index(), m_neuronIndex, m_neuronInputIndex, m_inputIndex,
        m_neuronDataDependency->get_operation_index(), *m_inputPastIndex,
        m_neuronDataDependency->get_value(*m_inputPastIndex), m_weightIndex,
        m_network.weight_table(m_weightIndex), weighted_input);
  }

  /* calculate u(x) part, u(x) is either the inputs starting from the next, or
   * the bias value(s) */
  if (m_isNextDepBias.has_value()) {
    RFASSERT(static_cast<bool>(m_nextDependency));
    RFASSERT(m_nextDependency->is_value_processed());
    const double next_value = m_nextDependency->get_value(0u /*past_index*/);
    RFASSERT_LOG("operation[{}]: Neuron[{}] Input[{}] u_x = {} op[{}]({})",
                 get_operation_index(), m_neuronIndex, m_neuronInputIndex,
                 *m_isNextDepBias ? "neuron input" : "bias",
                 m_neuronInputDependency->get_operation_index(), next_value);

    /* calculate the overall value */
    set_value(rafko_net::InputFunction::collect(get_input_function(),
                                                weighted_input, next_value));
    RFASSERT_LOG("operation[{}]: Neuron[{}] Input[{}] = {} (collected with {})",
                 get_operation_index(), m_neuronIndex, m_neuronInputIndex,
                 get_value(0u /*past_index*/),
                 Input_functions_Name(get_input_function()));

  } else {
    set_value(weighted_input);
    RFASSERT_LOG(
        "operation[{}]: Neuron[{}] Input[{}] = {} (no next dependency)",
        get_operation_index(), m_neuronIndex, m_neuronInputIndex,
        get_value(0u /*past_index*/));
  }
  set_value_processed();
}

void RafkoBackpropNeuronInputOperation::calculate_derivative(
    std::uint32_t d_w_index, const std::vector<double> &network_input,
    const std::vector<double> & /*label_data*/
) {
  RFASSERT(is_value_processed());
  RFASSERT(are_dependencies_registered());
  /* i(w) = w * f(w) ¤ u(w) | f(w) = network_input or internal_neuron_input */
  /* calculate f(x) part */
  double f_x_value;
  double f_x_derivative = 0.0;
  if (!m_inputPastIndex.has_value()) {
    /*!Note: Network inputs have no past value */
    f_x_value = get_value(0u /*past_index*/);
    if (m_weightIndex == d_w_index)
      f_x_derivative = network_input[m_inputIndex];
    RFASSERT_LOG("derivative_operation[{}](w[{}]): Neuron[{}] Input[{}]_d f_x "
                 "= input[{}]({}) * weight[{}]({}) = {}; f_x_d = {}",
                 get_operation_index(), d_w_index, m_neuronIndex,
                 m_neuronInputIndex, m_inputIndex, network_input[m_inputIndex],
                 m_weightIndex, m_network.weight_table(m_weightIndex),
                 f_x_value, f_x_derivative);
  } else { /* f(x) comes from Neuron data, may have inputs from the past */
    RFASSERT(static_cast<bool>(m_neuronDataDependency));
    RFASSERT((0u < *m_inputPastIndex) ||
             (m_neuronDataDependency->is_processed()));
    f_x_value = m_neuronDataDependency->get_value(*m_inputPastIndex);
    f_x_derivative =
        (m_neuronDataDependency->get_derivative(*m_inputPastIndex, d_w_index) *
         m_network.weight_table(m_weightIndex));
    if (m_weightIndex == d_w_index) {
      f_x_derivative += f_x_value;
      RFASSERT_LOG(
          "derivative_operation[{}](w[{}]): Neuron[{}] Input[{}]_d f_x = {}; "
          "f_x_d = {} = ({}(d_op[{}]) * {}(weight[{}])) + f_x",
          get_operation_index(), d_w_index, m_neuronIndex, m_neuronInputIndex,
          f_x_value, f_x_derivative,
          m_neuronDataDependency->get_derivative(*m_inputPastIndex, d_w_index),
          m_neuronDataDependency->get_operation_index(),
          m_network.weight_table(m_weightIndex), m_weightIndex);
    } else {
      RFASSERT_LOG(
          "derivative_operation[{}](w[{}]): Neuron[{}] Input[{}]_d f_x = {}; "
          "f_x_d = {} = ({}(d_op[{}]) * {}(weight[{}]))",
          get_operation_index(), d_w_index, m_neuronIndex, m_neuronInputIndex,
          f_x_value, f_x_derivative,
          m_neuronDataDependency->get_derivative(*m_inputPastIndex, d_w_index),
          m_neuronDataDependency->get_operation_index(),
          m_network.weight_table(m_weightIndex), m_weightIndex);
    }
  } /*if(is_network_input)*/

  /* calculate u(x) part, u(x) is either the inputs starting from the next, or
   * the bias value(s) */
  RFASSERT(static_cast<bool>(m_nextDependency));
  RFASSERT(m_nextDependency->is_processed());
  double u_x_value = 0.0;
  double u_x_derivative = 0.0;
  if (m_isNextDepBias.has_value() && *m_isNextDepBias) {
    u_x_value = m_nextDependency->get_value(0u /*past_index*/);
    u_x_derivative =
        m_nextDependency->get_derivative(0u /*past_index*/, d_w_index);
  } else if (m_isNextDepBias
                 .has_value()) { /* the last input starts to collect bias */
    u_x_value = m_nextDependency->get_value(0u /*past_index*/);
    u_x_derivative =
        m_nextDependency->get_derivative(0u /*past_index*/, d_w_index);
  }
  RFASSERT_LOG("derivative_operation[{}](w[{}]): Neuron[{}] Input[{}]_d u_x "
               "= {}(op[{}]); u_x_d = {}(d_op[{}]){}",
               get_operation_index(), d_w_index, m_neuronIndex,
               m_neuronInputIndex, u_x_value,
               m_nextDependency->get_operation_index(), u_x_derivative,
               m_nextDependency->get_operation_index(),
               (m_isNextDepBias.has_value())
                   ? (*m_isNextDepBias ? "(next dep)" : "(bias)")
                   : "(no next dependency)");

  /* calculate the derivative part */
  if (m_isNextDepBias.has_value()) {
    set_derivative(d_w_index, rafko_net::InputFunction::get_derivative(
                                  get_input_function(), f_x_value,
                                  f_x_derivative, u_x_value, u_x_derivative));
    RFASSERT_LOG("derivative operation[{}](w[{}]): Neuron[{}] Input[{}]_d = {} "
                 "(calculated with {})",
                 get_operation_index(), d_w_index, m_neuronIndex,
                 m_neuronInputIndex, get_derivative(d_w_index),
                 Input_functions_Name(get_input_function()));
  } else {
    set_derivative(d_w_index, f_x_derivative);
    RFASSERT_LOG("derivative operation[{}](w[{}]): Neuron[{}] Input[{}]_d = {} "
                 "(no next dependency)",
                 get_operation_index(), d_w_index, m_neuronIndex,
                 m_neuronInputIndex, f_x_derivative);
  }
  set_derivative_processed();
}

#if (RAFKO_USES_OPENCL)
std::string
RafkoBackpropNeuronInputOperation::local_declaration_operation() const {
  return R"(
    // Neuron input operation locals
    double f_x_value;
    double u_x_value;
    double f_x_derivative;
    double u_x_derivative;
  )";
}

std::string RafkoBackpropNeuronInputOperation::generic_value_kernel_operation(
    std::string network_input_array, std::string weight_array,
    std::string operations_value_array, std::string operations_array_size,
    std::string behavior_index) {
  std::string kernel_source = R"(
    // Calculate the weighted input(f(x))
    if(==past_index== == 0xFFu){ // past index at maximum means the input arrives from the network inputs
        f_x_value = ==neuron_input_array==[==f_x_op_index==] * ==weight_array==[==this_op_weight_index==];
    }else{
      if(==past_index== <= available_memory_slots){ // This is always true in case of Network inputs
          f_x_value = (
            ==f_x_value_array==[(long int)(==f_x_op_index==) - (long int)(==op_value_array_size== * ==past_index==) ]
            * ==weight_array==[==this_op_weight_index==]
          );
      }else{
        f_x_value = 0.0;
      }
    }

    // there is a next dependency
    if(==u_x_op_index== != 0xFFFFFFFFu){
        u_x_value = ==op_value_array==[==u_x_op_index==];
        ==input_kernel==
    }else{
      ==op_value_array==[==op_index==] = f_x_value;
    }

  )";

  /* Replacing the tokens with actual kernel string values */
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==input_kernel=="),
      rafko_net::InputFunction::get_all_kernel_value_functions(
          behavior_index, "==op_value_array==[==op_index==]", "f_x_value",
          "u_x_value"));
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==neuron_input_array=="), network_input_array);
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==f_x_value_array=="), operations_value_array);
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==weight_array=="), weight_array);
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==op_value_array=="), operations_value_array);
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==op_value_array_size=="),
      operations_array_size);
  return kernel_source;
}

std::string
RafkoBackpropNeuronInputOperation::generic_derivative_kernel_operation(
    std::string network_input_array, std::string weight_array,
    std::string operations_value_array, std::string operations_derivative_array,
    std::string operations_array_size, std::string behavior_index) {
  std::string kernel_source = R"(
    if(==past_index== == 0xFFu){ // past index at maximum means the input arrives from the network inputs
      f_x_value = ==op_value_array==[==op_index==];
      if(d_w_index == ==this_op_weight_index==){
        f_x_derivative = ==network_input_array==[==f_x_op_index==];
      }else{
        f_x_derivative = 0.0;
      }
    }else{ // otherwise input source is internal neuron data
      if(==past_index== <= available_memory_slots){
        f_x_value = ==op_value_array==[(long int)(==f_x_op_index==) - (long int)(==op_array_size== * ==past_index==)];
        f_x_derivative = (
          ==op_derivative_array==[(long int)(==f_x_op_index==) - (long int)(==op_array_size== * ==past_index==)]
          * ==weight_array==[==this_op_weight_index==]
        );
        if(==this_op_weight_index== == d_w_index){
          f_x_derivative += f_x_value;
        }
      }else{
        f_x_value = 0.0;
        f_x_derivative = 0.0;
      }
    }

    // there is a next dependency
    if(==u_x_op_index== != 0xFFFFFFFFu){
        u_x_value = ==op_value_array==[==u_x_op_index==];
        u_x_derivative = ==op_derivative_array==[==u_x_op_index==];
        ==input_kernel==
    }else{
      ==op_derivative_array==[==op_index==] = f_x_derivative;
    }
  )";

  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==input_kernel=="),
      rafko_net::InputFunction::get_all_kernel_derivative_functions(
          behavior_index, "==op_derivative_array==[==op_index==]", "f_x_value",
          "f_x_derivative", "u_x_value", "u_x_derivative"));
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==network_input_array=="),
      network_input_array);
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==weight_array=="), weight_array);
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==op_value_array=="), operations_value_array);
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==op_array_size=="), operations_array_size);
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==op_derivative_array=="),
      operations_derivative_array);
  return kernel_source;
}
#endif /*(RAFKO_USES_OPENCL)*/

} /* namespace rafko_gym */

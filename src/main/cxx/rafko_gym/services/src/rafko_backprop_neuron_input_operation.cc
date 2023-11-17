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
#include <stdexcept>

namespace {
const auto &s_isNeuronInputFromNetworkInput =
    rafko_net::SynapseIterator<rafko_net::InputSynapseInterval>::is_index_input;
const auto &s_convertToArrayIndex = rafko_net::SynapseIterator<
    rafko_net::InputSynapseInterval>::array_index_from_external_index;
} // namespace

namespace rafko_gym {

RafkoBackpropNeuronInputOperation::ConstructKit
RafkoBackpropNeuronInputOperation::calculate_current_operation_index_values(
    const rafko_net::RafkoNet &network, std::uint32_t neuron_index,
    std::uint32_t input_synapse_index, std::uint32_t weight_synapse_index,
    std::uint32_t start_in_input_synapse,
    std::uint32_t start_in_weight_synapse) {
  RFASSERT_LOG("Constructing Neuron input operation...");
  const rafko_net::Neuron &neuron = network.neuron_array(neuron_index);

  RFASSERT(static_cast<std::int32_t>(input_synapse_index) <
           neuron.input_indices_size());
  RFASSERT(static_cast<std::int32_t>(weight_synapse_index) <
           neuron.input_weights_size());
  const auto &input_synapse = neuron.input_indices(input_synapse_index);
  const auto &weight_synapse = neuron.input_weights(weight_synapse_index);

#ifndef NDEBUG
  rafko_net::SynapseIterator<rafko_net::InputSynapseInterval> inputs_iterator(
      neuron.input_indices());
  rafko_net::SynapseIterator<> weights_iterator(neuron.input_weights());
  RFASSERT_LOG("Input synapse is at: {}(index {}/{}) + {} --> {}",
               inputs_iterator.interval_starts_at(input_synapse_index),
               input_synapse_index, neuron.input_indices_size(),
               start_in_input_synapse,
               inputs_iterator.interval_starts_at(input_synapse_index) +
                   start_in_input_synapse);
  RFASSERT_LOG("Weight synapse is at: {}(index {}/{}) + {} --> {}",
               weights_iterator.interval_starts_at(weight_synapse_index),
               weight_synapse_index, neuron.input_weights_size(),
               start_in_weight_synapse,
               weights_iterator.interval_starts_at(weight_synapse_index) +
                   start_in_weight_synapse);
  if ((inputs_iterator.interval_starts_at(input_synapse_index) +
       start_in_input_synapse + 1u) !=
      (weights_iterator.interval_starts_at(weight_synapse_index) +
       start_in_weight_synapse)) {
    /*!Note: Weight index need be one after the input because of the spike
     * weight at the start of the Neuron*/
    throw std::runtime_error(
        "Input[" +
        std::to_string(
            (inputs_iterator.interval_starts_at(input_synapse_index) +
             start_in_input_synapse)) +
        "] and Weight[" +
        std::to_string(
            (weights_iterator.interval_starts_at(weight_synapse_index) +
             start_in_weight_synapse)) +
        "] synapse index values don't match to the same input!");
  }
#endif /*ifndef NDEBUG*/

  const std::optional<std::uint32_t> input_past_index =
      s_isNeuronInputFromNetworkInput(input_synapse.starts())
          ? std::nullopt /* Network inputs have no past reach */
          : std::optional<std::uint32_t>{input_synapse.reach_past_loops()};
  const std::int32_t input_synapse_start_index =
      (input_past_index.has_value())
          ? static_cast<std::uint32_t>(input_synapse.starts())
          : s_convertToArrayIndex(input_synapse.starts());
  RFASSERT_LOG("Starting input index: {} + {}(/{}) = {}",
               input_synapse_start_index, start_in_input_synapse,
               (input_synapse_start_index + input_synapse.interval_size()),
               start_in_input_synapse);
  RFASSERT(start_in_input_synapse < input_synapse.interval_size());
  const std::uint32_t starting_input_index =
      input_synapse_start_index + start_in_input_synapse;

  RFASSERT_LOG("Starting weight index: {} + {}(/{}) = {}",
               weight_synapse.starts(), start_in_weight_synapse,
               weight_synapse.interval_size(),
               weight_synapse.starts() + start_in_weight_synapse);
  RFASSERT(start_in_weight_synapse < weight_synapse.interval_size());
  const std::uint32_t starting_weight_index =
      weight_synapse.starts() + start_in_weight_synapse;

  const std::uint32_t current_span =
      std::min((input_synapse_start_index + input_synapse.interval_size() -
                starting_input_index),
               (weight_synapse.starts() + weight_synapse.interval_size() -
                starting_weight_index));
  RFASSERT_LOG("Current span: min(({} + {} - {}) , ({} + {} - {})) == {}",
               input_synapse_start_index, input_synapse.interval_size(),
               starting_input_index, weight_synapse.starts(),
               weight_synapse.interval_size(), starting_weight_index,
               current_span);
  RFASSERT_LOG("Current span check: inputs: {} + span / {} + {}; weights: {} + "
               "span / {} + {}",
               starting_input_index, input_synapse_start_index,
               input_synapse.interval_size(), starting_weight_index,
               weight_synapse.starts(), weight_synapse.interval_size());
  RFASSERT((starting_input_index + current_span) <=
           (input_synapse_start_index + input_synapse.interval_size()));
  RFASSERT((starting_weight_index + current_span) <=
           (weight_synapse.starts() + weight_synapse.interval_size()));

  /* If there are more weight synapses, or current is unfinished */
  std::optional<std::uint32_t> next_weight_synapse_index = std::nullopt;
  std::optional<std::uint32_t> next_start_index_in_next_weight_synapse =
      std::nullopt;
  RFASSERT_LOG("Checking if there are more weight synapses to cover: ({} + "
               "{}) < ({} + {}); {} < ({} - 1)",
               starting_weight_index, current_span, weight_synapse.starts(),
               weight_synapse.interval_size(), weight_synapse_index,
               neuron.input_weights_size());
  if ((starting_weight_index + current_span) <
      (weight_synapse.starts() + weight_synapse.interval_size())) {
    RFASSERT_LOG("next weight synapse: [{}] ({} + {} - {})",
                 weight_synapse_index, starting_weight_index, current_span,
                 weight_synapse.starts());
    next_weight_synapse_index.emplace(weight_synapse_index);
    next_start_index_in_next_weight_synapse.emplace(
        (starting_weight_index + current_span - weight_synapse.starts()));
  } else if (static_cast<std::int32_t>(weight_synapse_index) <
             (neuron.input_weights_size() - 1)) {
    RFASSERT_LOG("next weight synapse: [{} + 1] start", weight_synapse_index);
    next_weight_synapse_index.emplace(weight_synapse_index + 1);
    next_start_index_in_next_weight_synapse.emplace(0u);
  }

  /* If there are more input synapses, or current is unfinished */
  RFASSERT_LOG("Checking if there are more input synapses to cover: ({} + "
               "{}) < ({} + {}) || {} < ({} - 1)",
               starting_input_index, current_span, input_synapse_start_index,
               input_synapse.interval_size(), input_synapse_index,
               neuron.input_indices_size());
  std::optional<std::uint32_t> next_input_synapse_index = std::nullopt;
  std::optional<std::uint32_t> next_start_index_in_next_input_synapse =
      std::nullopt;
  if ((starting_input_index + current_span) <
      (input_synapse_start_index + input_synapse.interval_size())) {
    RFASSERT_LOG("next input synapse: [{}] ({} + {} - {})", input_synapse_index,
                 starting_input_index, current_span, input_synapse_start_index);
    next_input_synapse_index.emplace(input_synapse_index);
    next_start_index_in_next_input_synapse.emplace(
        (starting_input_index + current_span - input_synapse_start_index));
  } else if (static_cast<std::int32_t>(input_synapse_index) <
             (neuron.input_indices_size() - 1)) {
    RFASSERT_LOG("next input synapse: [{} + 1] start", input_synapse_index);
    next_input_synapse_index.emplace(input_synapse_index + 1);
    next_start_index_in_next_input_synapse.emplace(0u);
  }

  RFASSERT_LOG("next_input_synapse_index: {}",
               next_input_synapse_index.has_value()
                   ? std::to_string(next_input_synapse_index.value())
                   : "x");
  RFASSERT_LOG(
      "next_start_index_in_next_input_synapse: {}",
      next_start_index_in_next_input_synapse.has_value()
          ? std::to_string(next_start_index_in_next_input_synapse.value())
          : "x");
  RFASSERT((next_input_synapse_index.has_value() &&
            next_start_index_in_next_input_synapse.has_value()) ||
           (!next_input_synapse_index.has_value() &&
            !next_start_index_in_next_input_synapse.has_value()));
  RFASSERT_LOG("next_weight_synapse_index: {}",
               next_weight_synapse_index.has_value()
                   ? std::to_string(next_weight_synapse_index.value())
                   : "x");
  RFASSERT_LOG(
      "next_start_index_in_next_weight_synapse: {}",
      next_start_index_in_next_weight_synapse.has_value()
          ? std::to_string(next_start_index_in_next_weight_synapse.value())
          : "x");
  RFASSERT((next_weight_synapse_index.has_value() &&
            next_start_index_in_next_weight_synapse.has_value()) ||
           (!next_weight_synapse_index.has_value() &&
            !next_start_index_in_next_weight_synapse.has_value()));

  std::optional<SynapseSpan> next_span = std::nullopt;
  if (next_weight_synapse_index.has_value() &&
      next_start_index_in_next_weight_synapse.has_value()) {
    bool __workaround_isBias =
        (!next_input_synapse_index.has_value() &&
         !next_start_index_in_next_input_synapse.has_value() &&
         next_weight_synapse_index.has_value() &&
         next_start_index_in_next_weight_synapse.has_value());
    if (__workaround_isBias) {
      /* Bias operation should be based on the next weight index after the
       * current span */
      *next_weight_synapse_index =
          (weights_iterator.interval_starts_at(*next_weight_synapse_index) +
           start_in_weight_synapse + current_span);
    }
    next_span.emplace(SynapseSpan{
        next_input_synapse_index.value_or(0u), *next_weight_synapse_index,
        next_start_index_in_next_input_synapse.value_or(0u),
        *next_start_index_in_next_weight_synapse, __workaround_isBias});
  }
  return {current_span, starting_input_index, starting_weight_index,
          input_past_index, next_span};
}

RafkoBackpropNeuronInputOperation::RafkoBackpropNeuronInputOperation(
    RafkoBackpropagationData &data, const rafko_net::RafkoNet &network,
    std::uint32_t operation_index, std::uint32_t neuron_index,
    std::uint32_t input_synapse_index, std::uint32_t weight_synapse_index,
    std::uint32_t start_in_input_synapse, std::uint32_t start_in_weight_synapse)
    : RafkoBackpropNeuronInputOperation(
          data, network, operation_index, neuron_index,
          calculate_current_operation_index_values(
              network, neuron_index, input_synapse_index, weight_synapse_index,
              start_in_input_synapse, start_in_weight_synapse)) {}

RafkoBackpropNeuronInputOperation::RafkoBackpropNeuronInputOperation(
    RafkoBackpropagationData &data, const rafko_net::RafkoNet &network,
    std::uint32_t operation_index, std::uint32_t neuron_index, ConstructKit kit)
    : RafkoBackpropagationOperation(data, network, operation_index,
                                    ad_operation_neuron_input_d),
      m_neuronIndex(neuron_index), m_inputPastIndex(kit.m_inputPastIndex),
      m_nextOperation(kit.m_nextSpan),
      m_startingInputIndex(kit.m_startingInputIndex),
      m_startingWeightIndex(kit.m_startingWeightIndex),
      m_inputCount(kit.m_inputCount) {}

RafkoBackpropagationOperation::DependencyRequest
RafkoBackpropNeuronInputOperation::request_dependencies() {
  RFASSERT_LOG("Neuron input operation[{}]: request_dependencies called .. ",
               get_operation_index());
  RFASSERT(static_cast<std::int32_t>(m_neuronIndex) <
           m_network.neuron_array_size());
  RafkoBackpropagationOperation::DependencyParameters dependency_parameters;
  RFASSERT_LOG("Input past index: {}; ", m_inputPastIndex.has_value()
                                             ? std::to_string(*m_inputPastIndex)
                                             : "x");
  RFASSERT_LOG(
      "Next operation: {}",
      m_nextOperation.has_value()
          ? ("{" + std::to_string(m_nextOperation->m_inputSynapseIndex) + "," +
             std::to_string(m_nextOperation->m_weightSynapseIndex) + "," +
             std::to_string(m_nextOperation->m_startInInputSynapse) + "," +
             std::to_string(m_nextOperation->m_startInWeightSynapse) + "}")
          : "x");
  if (are_dependencies_registered() ||
      (!m_inputPastIndex.has_value() && !m_nextOperation.has_value()))
    return std::nullopt;

  RFASSERT_LOG("Neuron input operation[{}]: Requesting input dependencies..",
               get_operation_index());
  /* In case m_inputPastIndex has a value, this input operation takes its
   * input from internal Neuron data; In which case the required data is
   * collected from dependecies. In case a network input, the operation
   * doesn't require a dependency, and m_inputPastIndex doesn't have a value
   */
  if (m_inputPastIndex.has_value()) {
    RFASSERT_LOG("--> Collecting {} neuron data dependencies.. ", m_inputCount);
    for (std::uint32_t dependency_index = 0; dependency_index < m_inputCount;
         ++dependency_index) {
      RFASSERT(
          static_cast<std::int32_t>(m_startingInputIndex + dependency_index) <
          m_network.neuron_array_size());
      dependency_parameters.push_back(
          {ad_operation_neuron_spike_d,
           {static_cast<std::uint32_t>(m_startingInputIndex +
                                       dependency_index)}});
    }
  }

  if (m_nextOperation.has_value()) {
    RFASSERT_LOG("Neuron input operation[{}]: Requesting next operation ..",
                 get_operation_index());
    if (m_nextOperation.value().__workaround_m_isBias) {
      RFASSERT_LOG("--> Collecting Bias operation for neuron weight[{}]",
                   m_nextOperation->m_weightSynapseIndex);
      dependency_parameters.push_back(
          {ad_operation_neuron_bias_d,
           {m_neuronIndex, m_nextOperation->m_weightSynapseIndex}});
    } else {
      RFASSERT_LOG(
          "--> Collecting Succeding Neuron input operation: neuron index: "
          "{}; input synapse index: {}; weight synapse index: {}; "
          "start in input synapse: {}; start in weight synapse: {}",
          m_neuronIndex, m_nextOperation->m_inputSynapseIndex,
          m_nextOperation->m_weightSynapseIndex,
          m_nextOperation->m_startInInputSynapse,
          m_nextOperation->m_startInWeightSynapse);
      dependency_parameters.push_back(
          {ad_operation_neuron_input_d,
           {m_neuronIndex, m_nextOperation->m_inputSynapseIndex,
            m_nextOperation->m_weightSynapseIndex,
            m_nextOperation->m_startInInputSynapse,
            m_nextOperation->m_startInWeightSynapse}});
    }
  } else {
    RFASSERT_LOG("Neuron input operation[{}]: Has no next operation ..",
                 get_operation_index());
  }

  auto dependency_register_function =
      [this](std::vector<std::shared_ptr<RafkoBackpropagationOperation>>
                 dependencies) {
        std::uint32_t f_x_dependecy_count = 0u;
        if (m_inputPastIndex.has_value()) {
          f_x_dependecy_count = m_inputCount;
          RFASSERT(f_x_dependecy_count <= dependencies.size());
          RFASSERT(0 == m_neuronDataDependencies.size());
#if (RAFKO_USES_ASSERTLOGS)
          std::vector<std::uint32_t> collected_neuron_data;
          std::vector<std::uint32_t> collected_dependencies;
          collected_dependencies.reserve(m_inputCount);
          for (std::uint32_t dependency_index = 0;
               dependency_index < m_inputCount; ++dependency_index) {
            RFASSERT(static_cast<bool>(
                dependencies[dependency_index + dependency_index]));
            collected_neuron_data.push_back(m_startingInputIndex +
                                            dependency_index);
            collected_dependencies.push_back(
                dependencies[dependency_index]->get_operation_index());
          }
          RFASSERT_LOG(
              "Neuron input operation[{}]: Registering dependencies ..",
              get_operation_index());
          RFASSERT_LOGV(collected_neuron_data, "Neuron index values:");
          RFASSERT_LOGV(collected_dependencies, "Operation Index values:");
          RFASSERT_LOG("===================================================");
#endif /*(RAFKO_USES_ASSERTLOGS)*/

          m_neuronDataDependencies.insert(m_neuronDataDependencies.begin(),
                                          dependencies.begin(),
                                          dependencies.begin() + m_inputCount);
        }

        if (m_nextOperation.has_value()) {
          /* In case there's an f_x dependecy, the other dependency need be
           * after it */
          RFASSERT_LOG("Neuron input operation[{}]: f_x dependecy count: {}/{}",
                       get_operation_index(), f_x_dependecy_count,
                       dependencies.size());
          RFASSERT(f_x_dependecy_count < dependencies.size());
          RFASSERT(static_cast<bool>(dependencies[f_x_dependecy_count]));
          if (!m_nextOperation->__workaround_m_isBias) {
            RFASSERT_LOG(
                "Neuron input operation[{}]: Registering next dependency: "
                "operation[{}(?)] as other neuron input",
                get_operation_index(),
                dependencies[f_x_dependecy_count]->get_operation_index());
            m_nextInputDependency = dependencies[f_x_dependecy_count];
          } else {
            RFASSERT_LOG(
                "Neuron input operation[{}]: Registering next dependency: "
                "operation[{}(?)] as neuron bias",
                get_operation_index(),
                dependencies[f_x_dependecy_count]->get_operation_index());
            m_nextInputDependency = dependencies[f_x_dependecy_count];
          }
        }
        set_registered();
      };
  return {{dependency_parameters, dependency_register_function}};
}

std::vector<std::shared_ptr<RafkoBackpropagationOperation>>
RafkoBackpropNeuronInputOperation::get_own_dependencies() {
  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> dependencies;
  if (m_inputPastIndex.has_value() && (0 == m_inputPastIndex)) {
    dependencies.insert(dependencies.end(), m_neuronDataDependencies.begin(),
                        m_neuronDataDependencies.end());
  }
  if (m_nextInputDependency)
    dependencies.push_back(m_nextInputDependency);
  return dependencies;
}

std::vector<std::shared_ptr<RafkoBackpropagationOperation>>
RafkoBackpropNeuronInputOperation::get_own_dependencies_past_included() {
  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> dependencies;
  if (m_inputPastIndex.has_value()) {
    dependencies.insert(dependencies.end(), m_neuronDataDependencies.begin(),
                        m_neuronDataDependencies.end());
  }
  if (m_nextInputDependency)
    dependencies.push_back(m_nextInputDependency);
  RFASSERT(1u <= dependencies.size() || !are_dependencies_registered());
  return dependencies;
}

void RafkoBackpropNeuronInputOperation::calculate_value(
    const std::vector<double> &network_input) {
  RFASSERT(are_dependencies_registered());
  /* i(w) = w * f(w) ¤ u(w) | f(w) = network_input or internal_neuron_input */
  /* calculate f(x) part */
  double collected_value;
  if (m_inputPastIndex.has_value()) { /* f(x) comes from Neuron data */
    RFASSERT(0 < m_neuronDataDependencies.size());
    std::uint32_t input_index = 0u;
    for (const DependencyPointer &dependency : m_neuronDataDependencies) {
      RFASSERT(static_cast<bool>(dependency));
      RFASSERT((0u < *m_inputPastIndex) || (dependency->is_value_processed()));
      double weighted_input =
          (dependency->get_value(*m_inputPastIndex) *
           m_network.weight_table(m_startingWeightIndex + input_index));

      if (0 == input_index) {
        collected_value = weighted_input;
      } else {
        collected_value = rafko_net::InputFunction::collect(
            get_input_function(), weighted_input, collected_value);
      }

      ++input_index;
      RFASSERT_LOG("operation[{}]: Neuron[{}] Input[{}] f_x <-- "
                   "op[{}](past:{} = {}) * weight[{}]({}) = {}",
                   get_operation_index(), m_neuronIndex,
                   (m_startingInputIndex + input_index),
                   dependency->get_operation_index(), *m_inputPastIndex,
                   dependency->get_value(*m_inputPastIndex),
                   (m_startingWeightIndex + input_index),
                   m_network.weight_table(m_startingWeightIndex + input_index),
                   weighted_input);
    }
    RFASSERT(input_index == m_inputCount);
  } else { /* f(x) comes from network input */
    for (std::uint32_t input_index = 0; input_index < m_inputCount;
         ++input_index) {
      double weighted_input =
          network_input[m_startingInputIndex + input_index] *
          m_network.weight_table(m_startingWeightIndex + input_index);
      if (0 == input_index) {
        collected_value = weighted_input;
      } else {
        collected_value = rafko_net::InputFunction::collect(
            get_input_function(), weighted_input, collected_value);
      }
      RFASSERT_LOG(
          "operation[{}]: Neuron[{}] Input[{}] f_x <-- input[{}]({}) * "
          "weight[{}]({}) = {}",
          get_operation_index(), m_neuronIndex,
          (m_startingInputIndex + input_index),
          (m_startingInputIndex + input_index),
          network_input[(m_startingInputIndex + input_index)],
          (m_startingWeightIndex + input_index),
          m_network.weight_table(m_startingWeightIndex + input_index),
          weighted_input);
    }
  }

  if (static_cast<bool>(m_nextInputDependency)) {
    /* calculate u(x) part, u(x) is either the inputs starting from the next,
     * or the bias value(s) */
    RFASSERT(static_cast<bool>(m_nextInputDependency));
    RFASSERT(m_nextInputDependency->is_value_processed());
    double u_x_value = m_nextInputDependency->get_value(0u /*past_index*/);
    set_value(rafko_net::InputFunction::collect(get_input_function(), u_x_value,
                                                collected_value));
    RFASSERT_LOG("operation[{}]: Neuron[{}] Input result <-- "
                 "operation[{}]({}) = {}(collected with {})",
                 get_operation_index(), m_neuronIndex,
                 m_nextInputDependency->get_operation_index(),
                 m_nextInputDependency->get_value(0u /*past_index*/),
                 get_value(0u /*past_index*/),
                 Input_functions_Name(get_input_function()));
  } else {
    set_value(collected_value);
    RFASSERT_LOG("operation[{}]: Neuron[{}] Input result = {}(collected_value)",
                 get_operation_index(), m_neuronIndex, collected_value);
  }

  /* calculate the overall value */
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
  double collected_value;
  double collected_derivative;

  if (m_inputPastIndex.has_value()) { /* f(x) comes from Neuron data */
    RFASSERT(m_neuronDataDependencies.size() == m_inputCount);
    for (std::uint32_t input_index = 0; input_index < m_inputCount;
         ++input_index) {
      const std::uint32_t input_weight_index =
          m_startingWeightIndex + input_index;
      RFASSERT(input_index < m_neuronDataDependencies.size());
      const auto &dependency = m_neuronDataDependencies[input_index];
      RFASSERT(static_cast<bool>(dependency));
      RFASSERT((0u < *m_inputPastIndex) || (dependency->is_processed()));
      double f_x_value = dependency->get_value(*m_inputPastIndex);
      double f_x_derivative;
      if (input_weight_index == d_w_index) {
        f_x_derivative =
            (dependency->get_derivative(*m_inputPastIndex, d_w_index) *
             m_network.weight_table(input_weight_index));
        f_x_derivative += f_x_value;
      } else {
        f_x_derivative =
            dependency->get_derivative(*m_inputPastIndex, d_w_index);
      }

      if (0 == input_index) {
        collected_value = f_x_value;
        collected_derivative = f_x_derivative;
        RFASSERT_LOG("Derivative operation[{}](w[{}]): Neuron[{}] Input[{}]_d "
                     "first input",
                     get_operation_index(), d_w_index, m_neuronIndex,
                     input_index);
      } else {
        collected_value = f_x_value;
        collected_derivative = rafko_net::InputFunction::get_derivative(
            get_input_function(), f_x_value, f_x_derivative, collected_value,
            collected_derivative);
      }

      RFASSERT_LOG(
          "Derivative operation[{}](w[{}]): Neuron[{}] Input[{}]_d <-- "
          "(f_x_value: {}; (f_x_d: d_op[{}]({}) * weight[{}]({}){} = {}))",
          get_operation_index(), d_w_index, m_neuronIndex, input_index,
          f_x_value, dependency->get_operation_index(),
          dependency->get_derivative(*m_inputPastIndex, d_w_index),
          input_weight_index m_network.weight_table(input_weight_index),
          (input_weight_index == d_w_index) ? " + f_x_value" : "",
          f_x_derivative);
    }
  } else { /* f(x) comes from network input */
    for (std::uint32_t input_index = 0; input_index < m_inputCount;
         ++input_index) {
      const std::uint32_t input_weight_index =
          m_startingWeightIndex + input_index;
      const std::uint32_t input_input_index =
          m_startingInputIndex + input_index;
      double f_x_value = network_input[input_input_index] *
                         m_network.weight_table(input_weight_index);
      double f_x_derivative = ((input_weight_index == d_w_index)
                                   ? (network_input[input_input_index])
                                   : (0.0));

      if (0 == input_index) {
        collected_value = f_x_value;
        collected_derivative = f_x_derivative;
        RFASSERT_LOG("Derivative operation[{}](w[{}]): Neuron[{}] Input[{}]_d "
                     "first input",
                     get_operation_index(), d_w_index, m_neuronIndex,
                     input_index);
      } else {
        collected_value = rafko_net::InputFunction::get_derivative(
            get_input_function(), f_x_value, f_x_derivative, collected_value,
            collected_derivative);
      }

      RFASSERT_LOG(
          "Derivative operation[{}](w[{}]): Neuron[{}] Input[{}]_d "
          "<-- (value: input[{}]({}) * weight[{}]({}) = {}; derivative: {})",
          get_operation_index(), d_w_index, m_neuronIndex, input_index,
          input_input_index, network_input[input_input_index],
          input_weight_index, m_network.weight_table(input_weight_index),
          f_x_value, f_x_derivative);
    }
  }

  /* calculate u(x) part, u(x) is either the inputs starting from the next,
   * or the bias value(s) */
  double u_x_value = 0.0;
  double u_x_derivative = 0.0;
  if (static_cast<bool>(m_nextInputDependency)) {
    RFASSERT(m_nextOperation.has_value());
    RFASSERT(m_nextInputDependency->is_processed());
    u_x_value = m_nextInputDependency->get_value(0u /*past_index*/);
    u_x_derivative =
        m_nextInputDependency->get_derivative(0u /*past_index*/, d_w_index);
    RFASSERT_LOG("derivative_operation[{}](w[{}]): Neuron[{}] Input_d u_x "
                 "= op[{}]({}); u_x_d = d_op[{}]({}) (next {})",
                 get_operation_index(), d_w_index, m_neuronIndex,
                 m_nextInputDependency->get_operation_index(), u_x_value,
                 m_nextInputDependency->get_operation_index(), u_x_derivative,
                 (m_nextOperation->__workaround_m_isBias) ? "bias"
                                                          : "neuron_input");
  }

  /* calculate the derivative */
  set_derivative(d_w_index,
                 rafko_net::InputFunction::get_derivative(
                     get_input_function(), collected_value,
                     collected_derivative, u_x_value, u_x_derivative));
  RFASSERT_LOG("derivative operation[{}](w[{}]): Neuron[{}] Input_d = "
               "{}(calculated with value: {}; derivative: {})",
               get_operation_index(), d_w_index, m_neuronIndex, collected_value,
               collected_derivative,
               Input_functions_Name(get_input_function()));
  set_derivative_processed();
}

#if (RAFKO_USES_OPENCL)
std::string
RafkoBackpropNeuronInputOperation::local_declaration_operation() const {
  return R"(
    /* Neuron input operation locals */
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
  std::string operations = R"(
    // calculate the next value (u(x))
    u_x_value = ==op_value_array==[==u_x_op_index==];

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
  )";

  /* add the input function */
  operations += rafko_net::InputFunction::get_all_kernel_value_functions(
      behavior_index, "==op_value_array==[==op_index==]", "f_x_value",
      "u_x_value");

  /* Replacing the tokens with actual kernel string values */
  operations = rafko_utilities::replace_all_in_string(
      operations, std::regex("==neuron_input_array=="), network_input_array);
  operations = rafko_utilities::replace_all_in_string(
      operations, std::regex("==f_x_value_array=="), operations_value_array);
  operations = rafko_utilities::replace_all_in_string(
      operations, std::regex("==weight_array=="), weight_array);
  operations = rafko_utilities::replace_all_in_string(
      operations, std::regex("==op_value_array=="), operations_value_array);
  operations = rafko_utilities::replace_all_in_string(
      operations, std::regex("==op_value_array_size=="), operations_array_size);
  return operations;
}

std::string
RafkoBackpropNeuronInputOperation::generic_derivative_kernel_operation(
    std::string network_input_array, std::string weight_array,
    std::string operations_value_array, std::string operations_derivative_array,
    std::string operations_array_size, std::string behavior_index) {
  std::string kernel_source = R"(
    u_x_value = ==op_value_array==[==u_x_op_index==];
    u_x_derivative = ==op_derivative_array==[==u_x_op_index==];
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

    ==input_kernel==
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

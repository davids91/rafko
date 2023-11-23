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
#include "rafko_gym/services/rafko_autodiff_optimizer.hpp"

#include <deque>
#include <limits>

#include "rafko_gym/services/rafko_backprop_neuron_bias_operation.hpp"
#include "rafko_gym/services/rafko_backprop_neuron_input_operation.hpp"
#include "rafko_gym/services/rafko_backprop_objective_operation.hpp"
#include "rafko_gym/services/rafko_backprop_solution_feature_operation.hpp"
#include "rafko_gym/services/rafko_backprop_transfer_fn_operation.hpp"
#include "rafko_gym/services/rafko_backprop_weight_reg_operation.hpp"
#include "rafko_net/models/neuron_info.hpp"
#include "rafko_net/services/neuron_router.hpp"

namespace rafko_gym {

void RafkoAutodiffOptimizer::build(const std::shared_ptr<RafkoDataSet> data_set,
                                   std::shared_ptr<RafkoObjective> objective) {
  RFASSERT(static_cast<bool>(data_set));
  RFASSERT(static_cast<bool>(objective));
  m_usedSequenceTruncation = std::min(m_settings->get_memory_truncation(),
                                      data_set->get_sequence_size());
  m_usedMinibatchSize = std::min(m_settings->get_minibatch_size(),
                                 data_set->get_number_of_sequences());
  if (m_trainingEvaluator)
    m_trainingEvaluator->set_data_set(data_set);
  std::uint32_t w_relevant_op_count = build_without_data(data_set, objective);
  m_data.build(m_operations.size(), w_relevant_op_count,
               data_set->get_sequence_size());
  m_built = true;
}

std::uint32_t RafkoAutodiffOptimizer::build_without_data(
    const std::shared_ptr<RafkoDataSet> data_set,
    std::shared_ptr<RafkoObjective> objective) {
  RFASSERT_SCOPE(AUTODIFF_BUILD);
  RFASSERT(static_cast<bool>(data_set));
  RFASSERT(static_cast<bool>(objective));
  RFASSERT(m_unplacedSpikes.empty());
  RFASSERT(m_spikeSolvesFeatureMap.empty());
  std::uint32_t weight_relevant_operation_count;
  m_operations.clear();
  m_neuronIndexToSpikeOperationIndex = std::vector<std::uint32_t>(
      m_network.neuron_array_size(), s_NeuronNotYetAssigned);
  m_data.reset();
  if (m_trainingEvaluator) {
    m_trainingEvaluator->set_data_set(data_set);
    m_trainingEvaluator->set_objective(objective);
  }
  if (m_testEvaluator) {
    /* Test set should not be set inside this object */
    m_testEvaluator->set_objective(objective);
  }
  std::size_t neuron_count = 0;
  /*!Note: other components depend on the output objectives being the first
   * operations in the array. */
  for (std::uint32_t output_index = 0;
       output_index < m_network.output_neuron_number(); ++output_index) {
    m_operations.push_back(std::make_shared<RafkoBackpropObjectiveOperation>(
        m_data, m_network, *objective, m_operations.size(), output_index,
        data_set->get_number_of_label_samples()));
    RFASSERT_LOG("operation[{}]: {} for output {} ", m_operations.size() - 1,
                 Autodiff_operations_Name(ad_operation_objective_d),
                 output_index);
  }

  /* handle the group feature related operations, upload performance related
   * feature group operations. */
  std::uint32_t feature_group_index = 0u;
  for (const rafko_net::FeatureGroup &feature_group :
       m_network.neuron_group_features()) {
    if (rafko_net::NeuronInfo::is_feature_relevant_to_performance(
            feature_group.feature())) {
      m_operations.push_back(std::make_shared<RafkoBackpropWeightRegOperation>(
          *m_settings, m_data, m_network, m_operations.size(), feature_group));
      RFASSERT_LOG("operation[{}]: {} for feature_group[{}]",
                   m_operations.size() - 1,
                   Autodiff_operations_Name(
                       ad_operation_network_weight_regularization_feature),
                   feature_group_index);
    }
    ++feature_group_index;
  }
  /*!Note: weight_relevant_operation_count counts the items at the start of the
   * operation array */
  weight_relevant_operation_count = m_operations.size();

  /* Collect the Neuron subsets to determine the order of placement */
  rafko_net::NeuronRouter neuron_router(m_network);
  std::vector<std::deque<std::uint32_t>> neuron_subsets;
  bool strict_mode = false;
  while (!neuron_router.finished()) {
    neuron_router.collect_subset(m_settings->get_max_solve_threads(),
                                 m_settings->get_device_max_megabytes(),
                                 strict_mode);
    neuron_subsets.insert(neuron_subsets.begin(),
                          {neuron_router.get_subset().rbegin(),
                           neuron_router.get_subset().rend()});
    /*!Note: new subsets are inserted into the beginning of the array, so that
     * the Neurons depending on everything will be place to the beginning of the
     * operations array, as it is being executed from the end towards the
     * beginning. The deque is added with the reverse iterators, because in
     * non-strict mode order matters, so the neuron at the end might depend on
     * the other neurons, so it must be placed before them, in strict mode the
     * order doesn't matter.
     */

    for (const std::uint32_t &neuron_index :
         neuron_router
             .get_subset()) { /* confirm each Neuron as processed, and store the
                                 result in solved feature groups */
      std::vector<std::uint32_t> solved_features =
          neuron_router.confirm_first_subset_element_processed(neuron_index);
      for (std::uint32_t feature_group_index : solved_features) {
        if (rafko_net::NeuronInfo::is_feature_relevant_to_solution(
                m_network.neuron_group_features(feature_group_index)
                    .feature())) {
          m_spikeSolvesFeatureMap.insert({neuron_index, feature_group_index});
        }
      }                 /*for(each solved feature group index)*/
    }                   /*for(each neuron_index in the subset)*/
    strict_mode = true; /* non-strict mode should only run be enabled in the
                           first subset collection */
  }                     /*while(neuron_router is finished)*/

  RFASSERT_LOGV2(neuron_subsets, "Subset array:");

  /* Place one subset of Neurons */
  std::uint32_t done_index = 0;
  while (0u < neuron_subsets.size()) {
    for (std::uint32_t neuron_index : *neuron_subsets.begin()) {
      auto found_feature = m_spikeSolvesFeatureMap.find(neuron_index);
      if (found_feature != m_spikeSolvesFeatureMap.end()) {
        using FeaturePtr =
            std::shared_ptr<RafkoBackPropSolutionFeatureOperation>;
        FeaturePtr feature_operation =
            std::make_shared<RafkoBackPropSolutionFeatureOperation>(
                m_data, m_network, m_operations.size(), *m_settings,
                m_network.neuron_group_features(found_feature->second),
                m_neuronIndexToSpikeOperationIndex, m_executionThreads);
        m_operations.push_back(feature_operation);
        RFASSERT_LOG(
            "operation[{}]:  {} for feature_group[{}], triggered by Neuron[{}]",
            m_operations.size() - 1u,
            Neuron_group_features_Name(
                m_network.neuron_group_features(found_feature->second)
                    .feature()),
            found_feature->second, found_feature->first);
        for (std::uint32_t operation_index = 0;
             operation_index < (m_operations.size() - 1); ++operation_index) {
          m_operations[operation_index]->insert_dependency(feature_operation);
        }
        feature_operation->insert_dependency(
            place_spike_to_operations(neuron_index));
      } else
        place_spike_to_operations(neuron_index);
      /*!Note: Since the order of execution is backwards; The feature is to be
       * before the Spike operation. But apart from that, dependency needs to be
       * added explicitly to make it available for further paralell ordering
       * definition calculations
       */

      /* Upload dependencies for every operation until every dependency is
       * registered */
      while (done_index < m_operations.size()) {
        if (!m_operations[done_index]->are_dependencies_registered()) {
          RFASSERT_LOG("Registering dependencies for operation[{}]...",
                       done_index);
          RafkoBackpropagationOperation::DependencyRequest request =
              m_operations[done_index]->request_dependencies();
          if (request.has_value()) {
            auto &[parameters, dependency_register] = request.value();
            std::vector<std::shared_ptr<RafkoBackpropagationOperation>>
                new_dependencies;
            for (const RafkoBackpropagationOperation::DependencyParameter
                     &parameter : parameters) {
              new_dependencies.push_back(push_dependency(parameter));
            }
            dependency_register(new_dependencies);
          }
        }
        ++done_index;
      } /*while(done_index < operations.size())*/

    } /*for(every neuron_index in the collected subset)*/
    neuron_count += neuron_subsets.begin()->size();
    neuron_subsets.erase(neuron_subsets.begin());
  } /*while(subsets remain)*/

  RFASSERT(0 == std::count(m_neuronIndexToSpikeOperationIndex.begin(),
                           m_neuronIndexToSpikeOperationIndex.end(),
                           s_NeuronNotYetAssigned));
  RFASSERT_LOGV(m_neuronIndexToSpikeOperationIndex,
                "Spike Operation index for each Neuron:");
  return weight_relevant_operation_count;
}

void RafkoAutodiffOptimizer::calculate_value(
    const std::vector<double> &network_input) {
  for (std::int32_t operation_index = m_operations.size() - 1;
       operation_index >= 0; --operation_index) {
    m_operations[operation_index]->calculate_value(network_input);
  }
}

void RafkoAutodiffOptimizer::calculate_derivative(
    const std::vector<double> &network_input,
    const std::vector<double> &label_data) {
  m_executionThreads[0]->start_and_block([this, &network_input, &label_data](
                                             std::uint32_t thread_index) {
    const std::int32_t weights_in_one_thread =
        1 + (m_network.weight_table_size() /
             m_executionThreads[0]->get_number_of_threads());
    const std::int32_t weight_start_in_thread =
        (weights_in_one_thread * thread_index);
    const std::int32_t weights_to_do_in_this_thread =
        std::min(weights_in_one_thread,
                 (m_network.weight_table_size() - weight_start_in_thread));
    for (std::int32_t weight_index = weight_start_in_thread;
         weight_index < (weight_start_in_thread + weights_to_do_in_this_thread);
         ++weight_index) {
      for (std::int32_t operation_index = m_operations.size() - 1;
           operation_index >= 0; --operation_index)
        m_operations[operation_index]->calculate_derivative(
            static_cast<std::uint32_t>(weight_index), network_input,
            label_data);
    }
  });
}

void RafkoAutodiffOptimizer::calculate(BackpropDataBufferRange network_input,
                                       BackpropDataBufferRange label_data) {
  RFASSERT_SCOPE(AUTODIFF_CALCULATE);
  for (std::uint32_t run_index = 0; run_index < network_input.size();
       ++run_index) {
    m_data.step();
    calculate_value(network_input[run_index]);
    calculate_derivative(network_input[run_index], label_data[run_index]);
  }
}

void RafkoAutodiffOptimizer::update_context_errors(bool force_gpu_upload) {
  if ((m_trainingEvaluator) &&
      (0 == (m_iteration % m_settings->get_training_relevant_loop_count()))) {
    m_trainingEvaluator->refresh_solution_weights();
    m_lastTrainingError =
        -m_trainingEvaluator->stochastic_evaluation(force_gpu_upload);
  }
  if ((m_testEvaluator) &&
      ((m_iteration > (m_lastTestedIteration +
                       m_settings->get_training_relevant_loop_count())) ||
       ((m_trainingEvaluator) &&
        ((m_lastTestingError * m_settings->get_delta()) <
         std::abs(m_lastTrainingError - m_lastTestingError))))) {
    m_testEvaluator->refresh_solution_weights();
    m_lastTestingError =
        -m_testEvaluator->stochastic_evaluation(force_gpu_upload);
    m_lastTestedIteration = m_iteration;
  }
}

void RafkoAutodiffOptimizer::iterate(const RafkoDataSet &data_set,
                                     bool force_gpu_upload) {
  RFASSERT_SCOPE(AUTODIFF_ITERATE);
  std::uint32_t sequence_start_index =
      (rand() % (data_set.get_number_of_sequences() - m_usedMinibatchSize + 1));
  std::uint32_t start_index_inside_sequence =
      (rand() % (/* If the memory is truncated for the training.. */
                 data_set.get_sequence_size() - m_usedSequenceTruncation +
                 1u  /* ..not all result output values are evaluated.. */
                 )); /* ..only settings.get_memory_truncation(), starting at a
                        random index inside bounds */

  for (std::uint32_t sequence_index = sequence_start_index;
       sequence_index < m_usedMinibatchSize; ++sequence_index) {
    std::uint32_t raw_inputs_index =
        sequence_index *
        (data_set.get_sequence_size() + data_set.get_prefill_inputs_number());
    std::uint32_t raw_labels_index =
        sequence_index * data_set.get_sequence_size();

    /* Evaluate the current sequence step by step */
    reset();
    for (std::uint32_t prefill_iterator = 0;
         prefill_iterator < data_set.get_prefill_inputs_number();
         ++prefill_iterator) {
      m_data.step();
      calculate_value(data_set.get_input_sample(raw_inputs_index));
      ++raw_inputs_index;
    } /* The first few inputs are there to set an initial state to the network
       */

    /* Solve the data and store the result after the inital "prefill" */
    for (std::uint32_t sequence_index = 0;
         sequence_index < data_set.get_sequence_size(); ++sequence_index) {
      m_data.step();
      m_data
          .set_weight_derivative_update(/* Add to the relevant derivatives only
                                           when truncation parameters match */
                                        (sequence_index >=
                                         start_index_inside_sequence) &&
                                        (sequence_index <
                                         (start_index_inside_sequence +
                                          m_usedSequenceTruncation)));
      calculate_value(data_set.get_input_sample(raw_inputs_index));
      calculate_derivative(data_set.get_input_sample(raw_inputs_index),
                           data_set.get_label_sample(raw_labels_index));
      ++raw_inputs_index;
      ++raw_labels_index;
    } /*for(relevant sequences)*/
  }

  std::fill(m_tmpAvgD.begin(), m_tmpAvgD.end(), 0.0);
  for (std::uint32_t past_sequence_index = start_index_inside_sequence;
       past_sequence_index <
       (start_index_inside_sequence + m_usedSequenceTruncation);
       ++past_sequence_index) {
    const std::vector<double> &sequence_derivative =
        m_data.get_average_derivative().get_element(past_sequence_index);
    std::transform(
        sequence_derivative.begin(), sequence_derivative.end(),
        m_tmpAvgD.begin(), m_tmpAvgD.begin(),
        [](const double &a, const double &b) { return (a + b) / 2.0; });
  }

  if (static_cast<std::int32_t>(m_tmpAvgD.size()) >
      std::count(m_tmpAvgD.begin(), m_tmpAvgD.end(), 0.0))
    apply_weight_update(m_tmpAvgD);

  ++m_iteration;
  update_context_errors(force_gpu_upload);
}

double RafkoAutodiffOptimizer::get_avg_gradient(std::uint32_t d_w_index) const {
  double sum = 0.0;
  double count = 0.0;
  for (std::uint32_t past_index = 0u; past_index < m_network.memory_size();
       ++past_index) {
    sum += std::abs(m_data.get_average_derivative(past_index, d_w_index));
    count += 1.0;
  }
  return sum / count;
}

std::shared_ptr<RafkoBackpropagationOperation>
RafkoAutodiffOptimizer::place_spike_to_operations(
    std::uint32_t neuron_index,
    std::vector<RafkoBackpropagationOperation::Dependency> dependencies) {
  /* find the Spike index in the not yet placed Neuron spikes */
  auto found_element = m_unplacedSpikes.find(neuron_index);
  if (found_element != m_unplacedSpikes.end()) {
    found_element->second->set_operation_index(m_operations.size());
    m_operations.push_back(found_element->second);
    m_unplacedSpikes.erase(found_element);
    RFASSERT_LOG("operation[{}]:  Neuron[{}] {} inserted from unplaced spikes",
                 m_operations.size() - 1u, neuron_index,
                 Autodiff_operations_Name(ad_operation_neuron_spike_d));
  } else {
    m_operations.push_back(std::make_shared<RafkoBackpropSpikeFnOperation>(
        m_data, m_network, m_operations.size(), neuron_index));
    RFASSERT_LOG(
        "operation[{}]:  Neuron[{}] {} built, because not found elsewhere",
        m_operations.size() - 1u, neuron_index,
        Autodiff_operations_Name(ad_operation_neuron_spike_d));
  }
  RFASSERT_LOG("operation[{}]:  Neuron[{}] connection is stored (vec size: {})",
               m_operations.size() - 1u, neuron_index,
               m_neuronIndexToSpikeOperationIndex.size());
  RFASSERT(neuron_index < m_neuronIndexToSpikeOperationIndex.size());
  m_neuronIndexToSpikeOperationIndex[neuron_index] = (m_operations.size() - 1u);

  /* Insert provided dependencies */
  for (RafkoBackpropagationOperation::Dependency &dep : dependencies) {
    m_operations.back()->insert_dependency(dep);
  }
  return m_operations.back();
}

std::shared_ptr<RafkoBackpropagationOperation>
RafkoAutodiffOptimizer::find_or_queue_spike(std::uint32_t neuron_index) {

  /* find the Spike index in the already placed Neuron spikes */
  RFASSERT(neuron_index < m_neuronIndexToSpikeOperationIndex.size());
  if (m_neuronIndexToSpikeOperationIndex[neuron_index] !=
      s_NeuronNotYetAssigned) {
    return m_operations[m_neuronIndexToSpikeOperationIndex[neuron_index]];
  }

  /* find the Spike index in the not yet placed Neuron spikes */
  auto found_element = m_unplacedSpikes.find(neuron_index);
  if (found_element != m_unplacedSpikes.end())
    return found_element->second;

  /* Neuron index was not found, so add it to the unplaced spikes */
  auto insertion = m_unplacedSpikes.insert(
      {/* with a dummy operation index which is to be set in @place_spike */
       neuron_index,
       std::make_shared<RafkoBackpropSpikeFnOperation>(
           m_data, m_network, 0u /*operation index*/, neuron_index)});
  RFASSERT_LOG("Neuron[{}] {} inserted into unplaced spikes", neuron_index,
               Autodiff_operations_Name(ad_operation_neuron_spike_d));
  RFASSERT(std::get<1>(insertion));
  return std::get<0>(insertion)->second;
}

std::shared_ptr<RafkoBackpropagationOperation>
RafkoAutodiffOptimizer::push_dependency(
    RafkoBackpropagationOperation::DependencyParameter arguments) {
  switch (std::get<0>(arguments)) {
  case ad_operation_neuron_spike_d:
    RFASSERT(1u == std::get<1>(arguments).size());
    return find_or_queue_spike(std::get<1>(arguments)[0]);
  case ad_operation_neuron_transfer_d:
    RFASSERT(1u == std::get<1>(arguments).size());
    RFASSERT_LOG("Created operation[{}]: {} for Neuron[{}]",
                 m_operations.size(),
                 Autodiff_operations_Name(std::get<0>(arguments)),
                 std::get<1>(arguments)[0]);
    return m_operations.emplace_back(
        std::make_shared<RafkoBackpropTransferFnOperation>(
            m_data, m_network, m_operations.size(), std::get<1>(arguments)[0],
            *m_settings));
  case ad_operation_neuron_input_d: {
    RFASSERT(2u == std::get<1>(arguments).size());
    auto op = std::make_shared<RafkoBackpropNeuronInputOperation>(
        m_data, m_network, m_operations.size(), std::get<1>(arguments)[0],
        std::get<1>(arguments)[1]);
    RFASSERT_LOG(
        "Created operation[{}]: {} for Neuron[{}] Neuron input[{}]: input[{}]",
        m_operations.size(), Autodiff_operations_Name(std::get<0>(arguments)),
        std::get<1>(arguments)[0], std::get<1>(arguments)[1], op->m_inputIndex);
    return m_operations.emplace_back(op);
  }
  case ad_operation_neuron_bias_d:
    RFASSERT(2u == std::get<1>(arguments).size());
    RFASSERT_LOG("Created operation[{}]: {} for Neuron[{}] weight_input[{}] ( "
                 "weight[{}] ) ",
                 m_operations.size(),
                 Autodiff_operations_Name(std::get<0>(arguments)),
                 std::get<1>(arguments)[0], std::get<1>(arguments)[1],
                 rafko_net::SynapseIterator<rafko_net::IndexSynapseInterval>(
                     m_network.neuron_array(std::get<1>(arguments)[0])
                         .input_weights())[std::get<1>(arguments)[1]]);
    return m_operations.emplace_back(
        std::make_shared<RafkoBackpropNeuronBiasOperation>(
            m_data, m_network, m_operations.size(), std::get<1>(arguments)[0],
            std::get<1>(arguments)[1]));
  case ad_operation_objective_d: /* Objective operations are placed manually to
                                    the beginning of the vector */
  case ad_operation_unknown:
  default:
    break;
  }
  return std::shared_ptr<RafkoBackpropagationOperation>();
}

} /* namespace rafko_gym */

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

#include "rafko_mainframe/services/rafko_cpu_context.hpp"

#include <math.h>

#include "rafko_protocol/training.pb.h"
#include "rafko_utilities/models/data_ringbuffer.hpp"
#include "rafko_net/models/neuron_info.hpp"
#include "rafko_net/services/solution_builder.hpp"
#include "rafko_gym/models/rafko_dataset_wrapper.hpp"
#include "rafko_gym/services/updater_factory.hpp"

#include "rafko_mainframe/services/rafko_dummies.hpp"

namespace rafko_mainframe{

RafkoCPUContext::RafkoCPUContext(
  rafko_net::RafkoNet& neural_network, std::shared_ptr<rafko_mainframe::RafkoSettings> settings,
  std::shared_ptr<rafko_gym::RafkoObjective> objective
)
: RafkoContext(settings)
, m_network(neural_network)
, m_networkSolution(rafko_net::SolutionBuilder(*m_settings).build(m_network))
, m_weightAdapter(m_network, *m_networkSolution, *m_settings)
, m_agent(rafko_net::SolutionSolver::Builder(m_networkSolution, *m_settings).build())
, m_environment(std::make_unique<RafkoDummyEnvironment>(m_network.input_data_size(), m_network.output_neuron_number()))
, m_objective(objective)
, m_weightUpdater(rafko_gym::UpdaterFactory::build_weight_updater(m_network, rafko_gym::weight_updater_default, *m_settings))
, m_neuronOutputsToEvaluate( /* For every thread, 1 sequence is evaluated.. */
  (m_settings->get_max_processing_threads() * m_environment->get_sequence_size() + 1u),
  std::vector<double>(m_network.output_neuron_number()) /* ..plus for the label errors one additional vector is needed */
)
, m_executionThreads(m_settings->get_max_processing_threads())
, m_usedSequenceTruncation( std::min(m_settings->get_memory_truncation(), m_environment->get_sequence_size()) )
, m_usedMinibatchSize( std::min(m_settings->get_minibatch_size(), m_environment->get_number_of_sequences()) )
{
  m_neuronOutputsToEvaluate.back().resize(m_environment->get_number_of_label_samples());
}

void RafkoCPUContext::set_environment(std::shared_ptr<rafko_gym::RafkoEnvironment> environment){
  RFASSERT_LOG("Setting environment in CPU context..");
  RFASSERT_LOG("Environment feature size: {} vs. Network output Neuron number: {}", environment->get_feature_size(), m_network.output_neuron_number());
  RFASSERT(environment->get_feature_size() == m_network.output_neuron_number());
  RFASSERT_LOG("Environment input size: {} vs. Network input size: {}", environment->get_input_size(), m_network.input_data_size());
  RFASSERT(environment->get_input_size() == m_network.input_data_size());
  m_environment.reset();
  m_environment = environment;
  std::uint32_t old_output_buffer_num = m_neuronOutputsToEvaluate.size();
  std::uint32_t new_output_buffer_num = m_settings->get_max_processing_threads() * m_environment->get_sequence_size() + 1u;
  m_neuronOutputsToEvaluate.resize(new_output_buffer_num);
  if(old_output_buffer_num < new_output_buffer_num){
    for(std::uint32_t buffer_index = old_output_buffer_num-1; buffer_index < new_output_buffer_num; ++buffer_index){
      m_neuronOutputsToEvaluate[buffer_index].resize(m_environment->get_feature_size());
    }
  }
  m_neuronOutputsToEvaluate.back().resize(m_environment->get_number_of_label_samples());
  m_usedSequenceTruncation = std::min(m_settings->get_memory_truncation(), m_environment->get_sequence_size());
  m_usedMinibatchSize = std::min(m_settings->get_minibatch_size(), m_environment->get_number_of_sequences());
}

double RafkoCPUContext::error_post_process(double raw_error, std::uint32_t labels_evaluated){
  double result_error = raw_error;
  double divisor = std::max(1u, labels_evaluated);
  for(const rafko_net::FeatureGroup& feature : m_network.neuron_group_features()){
    if(rafko_net::NeuronInfo::is_feature_relevant_to_performance(feature.feature())){
      result_error += m_agent->expose_executor().calculate_performance_relevant(
        feature, *m_settings, m_network
      );
    }
  }

  RFASSERT_LOG(
    "Error post process: raw error value: {}; error corrected with performance: {}; divisor: {}",
    raw_error, result_error, divisor
  );

  return result_error / divisor;
}

double RafkoCPUContext::evaluate(std::uint32_t sequence_start, std::uint32_t sequences_to_evaluate, std::uint32_t start_index_in_sequence, std::uint32_t sequence_truncation){
  RFASSERT_LOG(
    "Evaluating sequences in CPU context: {} + {} / {}; start index inside sequence: {}; sequence truncation: {} ",
    sequence_start, sequences_to_evaluate, m_environment->get_number_of_sequences(), start_index_in_sequence, sequence_truncation
  );
  RFASSERT(m_environment->get_number_of_sequences() >= (sequence_start + sequences_to_evaluate));
  RFASSERT(static_cast<bool>(m_objective));

  double error_sum = (0.0);
  m_agent->set_eval_mode(true);
  for(std::uint32_t sequence_index = sequence_start; sequence_index < (sequence_start + sequences_to_evaluate); sequence_index += m_settings->get_max_processing_threads()){
    m_executionThreads.start_and_block([this, sequence_index](std::uint32_t thread_index){
      if(m_environment->get_number_of_sequences() > (sequence_index + thread_index)){ /* See if the sequence index is inside bounds */
        /*!Note: This might happen because of the number of used threads might point to a grater index, than the number of sequences;
         * Which is mainly because of division remainder between number fo threads and the number of sequences
         * */
        /* Solve the sequence under sequence_index + thread_index */
        std::uint32_t raw_inputs_index = (sequence_index + thread_index) * (m_environment->get_sequence_size() + m_environment->get_prefill_inputs_number());

        /* Evaluate the current sequence step by step */
        for(std::uint32_t prefill_iterator = 0; prefill_iterator < m_environment->get_prefill_inputs_number(); ++prefill_iterator){
          (void)m_agent->solve(m_environment->get_input_sample(raw_inputs_index), (0 == prefill_iterator), thread_index);
          ++raw_inputs_index;
        } /* The first few inputs are there to set an initial state to the network */

        /* Solve the data and store the result after the inital "prefill" */
        for(std::uint32_t sequence_iterator = 0; sequence_iterator < m_environment->get_sequence_size(); ++sequence_iterator){
          rafko_utilities::ConstVectorSubrange<> neuron_output = m_agent->solve(
            m_environment->get_input_sample(raw_inputs_index),
            ( (0u == m_environment->get_prefill_inputs_number())&&(0u == sequence_iterator) ),
            thread_index
          );
          std::copy( /* copy the result to the eval array */
            neuron_output.begin(), neuron_output.end(),
            m_neuronOutputsToEvaluate[(thread_index * m_environment->get_sequence_size()) + sequence_iterator].begin()
          );
          ++raw_inputs_index;
        }/*for(relevant sequences)*/
      }/*if(thread index inside bounds)*/
    });

    RFASSERT_LOGV2(m_neuronOutputsToEvaluate, "Neuron outputs to evaluate: ");

    double error_part = m_objective->set_features_for_sequences( /* Upload results to the data set */
      *m_environment, m_neuronOutputsToEvaluate,
      0u/* neuron_buffer_index */, sequence_index, std::min(
        ((sequence_start + sequences_to_evaluate) - (sequence_index)),
        static_cast<std::uint32_t>(m_settings->get_max_processing_threads())
      )/* sequences_to_evaluate */,
      start_index_in_sequence, sequence_truncation, m_neuronOutputsToEvaluate.back()
    );
    error_sum += error_part;
  } /* for(sequence_index: sequence_start --> (sequence start + sequences_to_evaluate)) */

  return -error_post_process( error_sum, (sequences_to_evaluate * m_environment->get_sequence_size()) );
}

} /* namespace rafko_mainframe */

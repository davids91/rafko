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

#include "rafko_mainframe/services/rafko_cpu_context.h"

#include <math.h>

#include "rafko_protocol/training.pb.h"
#include "rafko_utilities/models/data_ringbuffer.h"
#include "rafko_net/models/neuron_info.h"
#include "rafko_net/services/solution_builder.h"
#include "rafko_gym/models/rafko_dataset_wrapper.h"
#include "rafko_gym/services/updater_factory.h"

#include "rafko_mainframe/services/rafko_assertion_logger.h"
#include "rafko_mainframe/services/rafko_dummies.h"

namespace rafko_mainframe{

RafkoCPUContext::RafkoCPUContext(rafko_net::RafkoNet& neural_network_, rafko_mainframe::RafkoSettings settings_)
: settings(settings_)
, network(neural_network_)
, network_solution(rafko_net::SolutionBuilder(settings).build(network))
, agent(rafko_net::SolutionSolver::Builder(*network_solution, settings).build())
, environment(std::make_unique<RafkoDummyEnvironment>(network.input_data_size(), network.output_neuron_number()))
, objective(std::make_unique<RafkoDummyObjective>())
, weight_updater(rafko_gym::UpdaterFactory::build_weight_updater(network, *network_solution, rafko_gym::weight_updater_default, settings))
, neuron_outputs_to_evaluate( /* For every thread, 1 sequence is evaluated.. */
  (settings.get_max_processing_threads() * environment->get_sequence_size() + 1u),
  std::vector<double>(network.output_neuron_number()) /* ..plus for the label errors one additional vector is needed */
),execution_threads(settings.get_max_processing_threads())
, used_sequence_truncation( std::min(settings.get_memory_truncation(), environment->get_sequence_size()) )
, used_minibatch_size( std::min(settings.get_minibatch_size(), environment->get_number_of_sequences()) )
{ neuron_outputs_to_evaluate.back().resize(environment->get_number_of_label_samples()); }

void RafkoCPUContext::set_environment(std::shared_ptr<rafko_gym::RafkoEnvironment> environment_){
  RFASSERT(environment_->get_feature_size() == network.output_neuron_number());
  RFASSERT(environment_->get_input_size() == network.input_data_size());
  environment.reset();
  environment = environment_;
  std::uint32_t old_output_buffer_num = neuron_outputs_to_evaluate.size();
  std::uint32_t new_output_buffer_num = settings.get_max_processing_threads() * environment->get_sequence_size() + 1u;
  neuron_outputs_to_evaluate.resize(new_output_buffer_num);
  if(old_output_buffer_num < new_output_buffer_num){
    for(std::uint32_t buffer_index = old_output_buffer_num-1; buffer_index < new_output_buffer_num; ++buffer_index){
      neuron_outputs_to_evaluate[buffer_index].resize(environment->get_feature_size());
    }
  }
  neuron_outputs_to_evaluate.back().resize(environment->get_number_of_label_samples());
  used_sequence_truncation = std::min(settings.get_memory_truncation(), environment->get_sequence_size());
  used_minibatch_size = std::min(settings.get_minibatch_size(), environment->get_number_of_sequences());
}

double RafkoCPUContext::error_post_process(double raw_error, std::uint32_t labels_evaluated){
  double result_error = raw_error;
  double divisor = std::max(1u, labels_evaluated);

  for(const rafko_net::FeatureGroup& feature : network.neuron_group_features()){
    if(rafko_net::NeuronInfo::is_feature_relevant_to_performance(feature.feature())){
      result_error += agent->expose_executor().calculate_performance_relevant(
        feature, settings, network
      );
    }
  }

  return result_error / divisor;
}

double RafkoCPUContext::evaluate(std::uint32_t sequence_start, std::uint32_t sequences_to_evaluate, std::uint32_t start_index_in_sequence, std::uint32_t sequence_truncation){
  RFASSERT(environment->get_number_of_sequences() >= (sequence_start + sequences_to_evaluate));

  double error_sum = (0.0);
  agent->set_eval_mode(true);
  for(std::uint32_t sequence_index = sequence_start; sequence_index < (sequence_start + sequences_to_evaluate); sequence_index += settings.get_max_processing_threads()){
    execution_threads.start_and_block([this, sequence_index](std::uint32_t thread_index){
      if(environment->get_number_of_sequences() > (sequence_index + thread_index)){ /* See if the sequence index is inside bounds */
        /*!Note: This might happen because of the number of used threads might point to a grater index, than the number of sequences;
         * Which is mainly because of division remainder between number fo threads and the number of sequences
         * */
        /* Solve the sequence under sequence_index + thread_index */
        std::uint32_t raw_inputs_index = (sequence_index + thread_index) * (environment->get_sequence_size() + environment->get_prefill_inputs_number());

        /* Evaluate the current sequence step by step */
        for(std::uint32_t prefill_iterator = 0; prefill_iterator < environment->get_prefill_inputs_number(); ++prefill_iterator){
          (void)agent->solve(environment->get_input_sample(raw_inputs_index), (0 == prefill_iterator), thread_index);
          ++raw_inputs_index;
        } /* The first few labels are there to set an initial state to the network */

        /* Solve the data and store the result after the inital "prefill" */
        for(std::uint32_t sequence_iterator = 0; sequence_iterator < environment->get_sequence_size(); ++sequence_iterator){
          rafko_utilities::ConstVectorSubrange<> neuron_output = agent->solve(
            environment->get_input_sample(raw_inputs_index),
            ( (0u == environment->get_prefill_inputs_number())&&(0u == sequence_iterator) ),
            thread_index
          );
          std::copy( /* copy the result to the eval array */
            neuron_output.begin(), neuron_output.end(),
            neuron_outputs_to_evaluate[(thread_index * environment->get_sequence_size()) + sequence_iterator].begin()
          );
          ++raw_inputs_index;
        }
      }
    });

    double error_part = objective->set_features_for_sequences( /* Upload results to the data set */
      *environment, neuron_outputs_to_evaluate,
      0u/* neuron_buffer_index */, sequence_index, std::min(
        ((sequence_start + sequences_to_evaluate) - (sequence_index)),
        static_cast<std::uint32_t>(settings.get_max_processing_threads())
      )/* sequences_to_evaluate */,
      start_index_in_sequence, sequence_truncation, neuron_outputs_to_evaluate.back()
    );
    error_sum += error_part;
  } /* for(sequence_index: sequence_start --> (sequence start + sequences_to_evaluate)) */

  return -error_post_process( error_sum, (sequences_to_evaluate * environment->get_sequence_size()) );
}

} /* namespace rafko_mainframe */

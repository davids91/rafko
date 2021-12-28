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

 #include "rafko_gym/services/rafko_environment_data_set.h"

#include <math.h>

#include "rafko_utilities/models/data_ringbuffer.h"

namespace rafko_gym{

RafkoEnvironmentDataSet::RafkoEnvironmentDataSet(
  rafko_mainframe::RafkoSettings& settings_,
  RafkoDatasetWrapper&& training_set_, RafkoDatasetWrapper&& test_set_, rafko_net::Cost_functions cost_function
):settings(settings_)
, training_set(training_set_)
, training_cost(settings, training_set, cost_function)
, test_set(test_set_)
, test_cost(settings, test_set, cost_function)
, neuron_outputs_to_evaluate( /* For every thread, 1 sequence is evaluated.. */
  (settings.get_max_processing_threads() * training_set.get_sequence_size() + 1u),
  std::vector<sdouble32>(training_set_.get_feature_size()) /* ..plus for the label errors one additional vector is needed */
)
, execution_threads(settings.get_max_processing_threads())
, loops_unchecked(settings.get_tolerance_loop_value() + 1u)
, used_sequence_truncation( std::min(settings.get_memory_truncation(), training_set.get_sequence_size()) )
{
  (void)settings.set_minibatch_size(std::max(1u,std::min(
    training_set.get_number_of_sequences(),settings.get_minibatch_size()
  )));
  (void)settings.set_memory_truncation(std::max(1u,std::min(
    training_set.get_sequence_size(), settings.get_memory_truncation()
  )));
  neuron_outputs_to_evaluate.back().resize(training_set.get_number_of_label_samples());
}

void RafkoEnvironmentDataSet::evaluate(
  RafkoAgent& agent, DataAggregate& cost_container, uint32 sequence_start, uint32 sequences_to_evaluate,
  uint32 start_index_in_sequence, uint32 sequence_truncation
){
  assert(cost_container.get_dataset().get_number_of_sequences() >= (sequence_start + sequences_to_evaluate));
  assert(cost_container.get_dataset().get_feature_size() == agent.get_solution().output_neuron_number());

  cost_container.expose_to_multithreading();
  for(uint32 sequence_index = sequence_start; sequence_index < (sequence_start + sequences_to_evaluate); sequence_index += settings.get_max_processing_threads()){
    execution_threads.start_and_block([this, &agent, &cost_container, sequence_index](uint32 thread_index){
      evaluate_single_sequence(agent, cost_container, sequence_index, thread_index);
    });
    cost_container.set_features_for_sequences( /* Upload results to the data set */
      neuron_outputs_to_evaluate, 0u,
      sequence_index, std::min(((sequence_start + sequences_to_evaluate) - (sequence_index)), static_cast<uint32>(settings.get_max_processing_threads())),
      start_index_in_sequence, sequence_truncation, neuron_outputs_to_evaluate.back()
    );
  } /* for(sequence_index: sequence_start --> (sequence start + sequences_to_evaluate)) */
  cost_container.conceal_from_multithreading();
}

void RafkoEnvironmentDataSet::evaluate_single_sequence(RafkoAgent& agent, DataAggregate& cost_container, uint32 sequence_index, uint32 thread_index){
  if(cost_container.get_dataset().get_number_of_sequences() > (sequence_index + thread_index)){ /* See if the sequence index is inside bounds */
    /*!Note: This might happen because of the number of used threads might point to a grater index, than the number of sequences;
     * Which is mainly because of division remainder between number fo threads and the number of sequences
     * */
    /* Solve the sequence under sequence_index + thread_index */
    uint32 raw_label_index = sequence_index + thread_index;
    uint32 raw_inputs_index = raw_label_index * (cost_container.get_dataset().get_sequence_size() + cost_container.get_dataset().get_prefill_inputs_number());
    raw_label_index *= cost_container.get_dataset().get_sequence_size();

    /* Evaluate the current sequence step by step */
    for(uint32 prefill_iterator = 0; prefill_iterator < cost_container.get_dataset().get_prefill_inputs_number(); ++prefill_iterator){
      (void)agent.solve(cost_container.get_dataset().get_input_sample(raw_inputs_index), (0 == prefill_iterator), thread_index);
      ++raw_inputs_index;
    } /* The first few labels are there to set an initial state to the network */

    /* Solve the data and store the result after the inital "prefill" */
    for(uint32 sequence_iterator = 0; sequence_iterator < cost_container.get_dataset().get_sequence_size(); ++sequence_iterator){
      bool reset = ( (0u == cost_container.get_dataset().get_prefill_inputs_number())&&(0u == sequence_iterator) );
      rafko_utilities::ConstVectorSubrange<> neuron_output = agent.solve(cost_container.get_dataset().get_input_sample(raw_inputs_index), reset, thread_index);
      std::copy( /* copy the result to the eval array */
        neuron_output.begin(), neuron_output.end(),
        neuron_outputs_to_evaluate[(thread_index * cost_container.get_dataset().get_sequence_size()) + sequence_iterator].begin()
      );
      ++raw_label_index;
      ++raw_inputs_index;
    }
  }
}

} /* namespace rafko_gym */

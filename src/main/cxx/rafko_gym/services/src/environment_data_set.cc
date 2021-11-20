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

#include "rafko_gym/services/environment_data_set.h"

#include <cmath>

#include "rafko_utilities/models/data_ringbuffer.h"

namespace rafko_gym{

using std::min;
using std::max;

using rafko_utilities::DataRingbuffer;

EnvironmentDataSet::EnvironmentDataSet(ServiceContext& service_context_, DataAggregate& train_set_, DataAggregate& test_set_)
: service_context(service_context_)
, train_set(train_set_)
, test_set(test_set_)
, neuron_outputs_to_evaluate( /* For every thread, 1 sequence is evaluated.. */
  (service_context.get_max_processing_threads() * train_set.get_sequence_size() + 1u),
  vector<sdouble32>(train_set_.get_feature_size()) /* ..plus for the label errors one additional vector is needed */
)
, execution_threads(service_context.get_max_processing_threads())
, loops_unchecked(service_context.get_tolerance_loop_value() + 1u)
, used_sequence_truncation(min(service_context.get_memory_truncation(), train_set.get_sequence_size()))
{
  (void)service_context.set_minibatch_size(max(1u,min(
    train_set.get_number_of_sequences(),service_context.get_minibatch_size()
  )));
  (void)service_context.set_memory_truncation(max(1u,min(
    train_set.get_sequence_size(), service_context.get_memory_truncation()
  )));
  neuron_outputs_to_evaluate.back().resize(train_set.get_number_of_label_samples());
}

void EnvironmentDataSet::evaluate(
  Agent& agent, DataAggregate& data_set, uint32 sequence_start, uint32 sequences_to_evaluate,
  uint32 start_index_in_sequence, uint32 sequence_truncation
){
  if(data_set.get_number_of_sequences() < (sequence_start + sequences_to_evaluate))
    throw std::runtime_error("Sequence interval out of bounds!");

  if(train_set.get_feature_size() != agent.get_solution().output_neuron_number())
    throw std::runtime_error("Network output size doesn't match size of provided labels!");

  data_set.expose_to_multithreading();
  for(uint32 sequence_index = sequence_start; sequence_index < (sequence_start + sequences_to_evaluate); sequence_index += service_context.get_max_processing_threads()){
    execution_threads.start_and_block([this, &agent, &data_set, sequence_index](uint32 thread_index){
      evaluate_single_sequence(agent, data_set, sequence_index, thread_index);
    });
    data_set.set_features_for_sequences( /* Upload results to the data set */
      neuron_outputs_to_evaluate, 0u,
      sequence_index, min(((sequence_start + sequences_to_evaluate) - (sequence_index)), static_cast<uint32>(service_context.get_max_processing_threads())),
      start_index_in_sequence, sequence_truncation, neuron_outputs_to_evaluate.back()
    );
  } /* for(sequence_index: sequence_start --> (sequence start + sequences_to_evaluate)) */
  data_set.conceal_from_multithreading();
}

void EnvironmentDataSet::evaluate_single_sequence(Agent& agent, DataAggregate& data_set, uint32 sequence_index, uint32 thread_index){
  if(data_set.get_number_of_sequences() > (sequence_index + thread_index)){ /* See if the sequence index is inside bounds */
    /*!Note: This might happen because of the number of used threads might point to a grater index, than the number of sequences;
     * Which is mainly because of division remainder between number fo threads and the number of sequences
     * */
    /* Solve the sequence under sequence_index + thread_index */
    uint32 raw_label_index = sequence_index + thread_index;
    uint32 raw_inputs_index = raw_label_index * (data_set.get_sequence_size() + data_set.get_prefill_inputs_number());
    raw_label_index *= data_set.get_sequence_size();

    /* Evaluate the current sequence step by step */
    for(uint32 prefill_iterator = 0; prefill_iterator < data_set.get_prefill_inputs_number(); ++prefill_iterator){
      (void)agent.solve(data_set.get_input_sample(raw_inputs_index), (0 == prefill_iterator), thread_index);
      ++raw_inputs_index;
    } /* The first few labels are there to set an initial state to the network */

    /* Solve the data and store the result after the inital "prefill" */
    for(uint32 sequence_iterator = 0; sequence_iterator < data_set.get_sequence_size(); ++sequence_iterator){
      bool reset = ( (0u == data_set.get_prefill_inputs_number())&&(0u == sequence_iterator) );
      const DataRingbuffer& neuron_output = agent.solve(data_set.get_input_sample(raw_inputs_index), reset, thread_index);
      std::copy( /* copy the result to the eval array */
        neuron_output.get_const_element(0).begin(),neuron_output.get_const_element(0).end(),
        neuron_outputs_to_evaluate[(thread_index * data_set.get_sequence_size()) + sequence_iterator].begin()
      );
      ++raw_label_index;
      ++raw_inputs_index;
    }
  }
}

} /* namespace rafko_gym */

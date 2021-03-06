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
#include "sparse_net_library/models/data_aggregate.h"

namespace sparse_net_library {

void Data_aggregate::fill(Data_set& samples){
  uint32 feature_start_index = 0;
  uint32 input_start_index = 0;
  /*!Note: One cycle can be used for both, because there will always be at least as many inputs as labels */
  for(uint32 raw_sample_iterator = 0; raw_sample_iterator < input_samples.size(); ++ raw_sample_iterator){
    input_samples[raw_sample_iterator] = vector<sdouble32>(samples.input_size());
    for(uint32 input_iterator = 0; input_iterator < samples.input_size(); ++input_iterator)
      input_samples[raw_sample_iterator][input_iterator] = samples.inputs(input_start_index + input_iterator);
    input_start_index += samples.input_size();
    if(raw_sample_iterator < label_samples.size()){
      label_samples[raw_sample_iterator] = vector<sdouble32>(samples.feature_size());
      for(uint32 feature_iterator = 0; feature_iterator < samples.feature_size(); ++feature_iterator)
        label_samples[raw_sample_iterator][feature_iterator] = samples.labels(feature_start_index + feature_iterator);
      feature_start_index += samples.feature_size();
    }
  }
}

void Data_aggregate::set_features_for_labels(
  const vector<vector<sdouble32>>& neuron_data,
  uint32 neuron_buffer_start_index, uint32 raw_start_index, uint32 labels_to_evaluate
){
  if((raw_start_index + labels_to_evaluate) <= error_state.back().sample_errors.size()){
    cost_function->get_feature_errors(
      label_samples, neuron_data, error_state.back().sample_errors,
      raw_start_index, labels_to_evaluate, neuron_buffer_start_index, get_number_of_label_samples()
    );

    error_state.back().error_sum = 0;
    for(uint32 sample_index = 0; sample_index < label_samples.size() ; ++sample_index)
      error_state.back().error_sum += error_state.back().sample_errors[sample_index];
  }else throw new std::runtime_error("Label index out of bounds!");
}

void Data_aggregate::set_features_for_labels(
  vector<unique_ptr<Solution_solver>>& network_solvers, uint32 label_start_index, uint32 labels_to_eval,
  uint32 start_index_in_sequence, uint32 sequence_truncation
){
  uint32 sequence_index = label_start_index;
  const uint32 sequences_to_evaluate = std::min(get_number_of_sequences(), labels_to_eval);
  const uint32 sequences_in_one_thread = 1 + static_cast<uint32>(sequences_to_evaluate/network_solvers.size());
  solve_threads.reserve(network_solvers.size());

  for(uint32 thread_index = 0; ((thread_index < network_solvers.size())&&(get_number_of_sequences() > sequence_index)); ++thread_index){
    solve_threads.push_back(thread( /* As long as there are threads to open for remaining weights, open threads */
      &Data_aggregate::set_features_for_labels_thread, this, ref(network_solvers), thread_index, sequence_index,
      std::min(sequences_in_one_thread, (get_number_of_sequences() - sequence_index)),
      start_index_in_sequence, sequence_truncation
    ));
    sequence_index += sequences_in_one_thread;
  }
  while(0 < solve_threads.size()){
    if(solve_threads.back().joinable()){
      solve_threads.back().join();
      solve_threads.pop_back();
    }
  }
}

void Data_aggregate::set_features_for_labels_thread(
    vector<unique_ptr<Solution_solver>>& network_solvers, uint32 solve_thread_index,
    uint32 sequence_start_index, uint32 sequences_to_evaluate,
    uint32 start_index_in_sequence, uint32 sequence_truncation
){
  uint32 raw_inputs_index;
  uint32 raw_label_index;

  for(uint32 sample = 0; sample < sequences_to_evaluate; ++sample){
    raw_label_index = sequence_start_index + sample;
    raw_inputs_index = raw_label_index * (get_sequence_size() + get_prefill_inputs_number());
    raw_label_index = raw_label_index * get_sequence_size();

    /* Prefill network with the initial inputs */
    network_solvers[solve_thread_index]->reset();
    for(uint32 prefill_iterator = 0; prefill_iterator < get_prefill_inputs_number(); ++prefill_iterator){
      network_solvers[solve_thread_index]->solve(get_input_sample(raw_inputs_index));
      ++raw_inputs_index;
    }

    /* Evaluate the current sequence step by step */
    for(uint32 sequence_iterator = 0; sequence_iterator < get_sequence_size(); ++sequence_iterator){
      network_solvers[solve_thread_index]->solve(get_input_sample(raw_inputs_index)); /* Solve the network for the sampled labels input */
      ++raw_label_index;
      ++raw_inputs_index;
    }
    std::lock_guard<mutex> my_lock(dataset_mutex);
    set_features_for_labels(
      network_solvers[solve_thread_index]->get_neuron_memory().get_whole_buffer(), start_index_in_sequence,
      ((sequence_start_index + sample) * get_sequence_size()) + start_index_in_sequence,
      sequence_truncation /* To avoid vanishing gradients with sequential data, error calculation is truncated */
    ); /* Re-calculate error for the training set */
  }
}

} /* namespace sparse_net_library */

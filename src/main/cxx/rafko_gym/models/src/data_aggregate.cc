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
#include "rafko_gym/models/data_aggregate.h"

#include <math.h>

namespace rafko_gym{

rafko_utilities::DataPool<sdouble32> DataAggregate::common_datapool(1,1);

void DataAggregate::fill(rafko_gym::DataSet& samples){
  uint32 feature_start_index = 0;
  uint32 input_start_index = 0;
  /*!Note: One cycle can be used for both, because there will always be at least as many inputs as labels */
  for(uint32 raw_sample_iterator = 0; raw_sample_iterator < input_samples.size(); ++ raw_sample_iterator){
    input_samples[raw_sample_iterator] = std::vector<sdouble32>(samples.input_size());
    for(uint32 input_iterator = 0; input_iterator < samples.input_size(); ++input_iterator)
      input_samples[raw_sample_iterator][input_iterator] = samples.inputs(input_start_index + input_iterator);
    input_start_index += samples.input_size();
    if(raw_sample_iterator < label_samples.size()){
      label_samples[raw_sample_iterator] = std::vector<sdouble32>(samples.feature_size());
      for(uint32 feature_iterator = 0; feature_iterator < samples.feature_size(); ++feature_iterator)
        label_samples[raw_sample_iterator][feature_iterator] = samples.labels(feature_start_index + feature_iterator);
      feature_start_index += samples.feature_size();
    }
  }
}

void DataAggregate::set_feature_for_label(uint32 sample_index, const std::vector<sdouble32>& neuron_data){
  if(label_samples.size() > sample_index){
    if(!exposed_to_multithreading){
      std::lock_guard<std::mutex> my_lock(dataset_mutex);
      error_state.back().error_sum -= error_state.back().sample_errors[sample_index];
    }
    error_state.back().sample_errors[sample_index] = cost_function->get_feature_error(
      label_samples[sample_index], neuron_data, get_number_of_label_samples()
    );
    if(!exposed_to_multithreading){
      std::lock_guard<std::mutex> my_lock(dataset_mutex);
      error_state.back().error_sum += error_state.back().sample_errors[sample_index];
    }
  }else throw std::runtime_error("Sample index out of bounds!");
}

void DataAggregate::set_features_for_labels(
  const std::vector<std::vector<sdouble32>>& neuron_data,
  uint32 neuron_buffer_start_index, uint32 raw_start_index, uint32 labels_to_evaluate
){
  if((raw_start_index + labels_to_evaluate) <= error_state.back().sample_errors.size()){
    cost_function->get_feature_errors(
      label_samples, neuron_data, error_state.back().sample_errors,
      raw_start_index, raw_start_index, labels_to_evaluate, neuron_buffer_start_index, get_number_of_label_samples()
    );
    if(!exposed_to_multithreading){
      error_state.back().error_sum = 0;
      error_calculation_threads.start_and_block(error_calculation_lambda);
    }
  }else throw std::runtime_error("Label index out of bounds!");
}

void DataAggregate::set_features_for_sequences(
  const std::vector<std::vector<sdouble32>>& neuron_data, uint32 neuron_buffer_start_index,
  uint32 sequence_start_index, uint32 sequences_to_evaluate,
  uint32 start_index_in_sequence, uint32 sequence_truncation,
  std::vector<sdouble32>& tmp_data
){
  if((sequence_start_index + sequences_to_evaluate) <= get_number_of_sequences()){
    if((start_index_in_sequence + sequence_truncation) > get_sequence_size())
      throw std::runtime_error("Sequence truncation indices out of bounds!");

    uint32 raw_start_index = sequence_start_index * get_sequence_size();
    uint32 labels_to_evaluate = sequences_to_evaluate * get_sequence_size();
    tmp_data.resize(labels_to_evaluate);

    cost_function->get_feature_errors(
      label_samples, neuron_data, tmp_data,
      raw_start_index, 0, labels_to_evaluate, neuron_buffer_start_index, get_number_of_label_samples()
    );
    uint32 copy_index_base = 0;
    for(uint32 sequence_iterator = 0; sequence_iterator < sequences_to_evaluate; ++sequence_iterator){
      std::copy(
        tmp_data.begin() + copy_index_base + start_index_in_sequence,
        tmp_data.begin() + copy_index_base + start_index_in_sequence + sequence_truncation,
        error_state.back().sample_errors.begin() + raw_start_index + copy_index_base + start_index_in_sequence
      );
      copy_index_base += get_sequence_size();
    }

    if(!exposed_to_multithreading){
      error_state.back().error_sum = 0;
      error_calculation_threads.start_and_block(error_calculation_lambda);
    }
  }else throw std::runtime_error("Sequence index out of bounds!");
}

void DataAggregate::conceal_from_multithreading(){
  exposed_to_multithreading = false;
  error_state.back().error_sum = 0;
  error_calculation_threads.start_and_block(error_calculation_lambda);
}

void DataAggregate::accumulate_error_sum(uint32 error_start, uint32 errors_to_sum){
  sdouble32 local_error = 0;
  for(uint32 sample_index = error_start; sample_index < (error_start + errors_to_sum) ; ++sample_index)
    local_error += error_state.back().sample_errors[sample_index];
  std::lock_guard<std::mutex> my_lock(dataset_mutex);
  error_state.back().error_sum += local_error;
}

} /* namespace rafko_gym */

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
#include "rafko_gym/models/rafko_dataset_cost.h"

#include <math.h>

namespace rafko_gym{

rafko_utilities::DataPool<sdouble32> RafkoDatasetCost::common_datapool(1,1);

void RafkoDatasetCost::set_feature_for_label(uint32 sample_index, const std::vector<sdouble32>& neuron_data){
  assert(dataset.get_number_of_label_samples() > sample_index);
  if(!exposed_to_multithreading){
    std::lock_guard<std::mutex> my_lock(dataset_mutex);
    error_state.back().error_sum -= error_state.back().sample_errors[sample_index];
  }
  error_state.back().sample_errors[sample_index] = cost_function->get_feature_error(
    dataset.get_label_sample(sample_index), neuron_data, dataset.get_number_of_label_samples()
  );
  if(!exposed_to_multithreading){
    std::lock_guard<std::mutex> my_lock(dataset_mutex);
    error_state.back().error_sum += error_state.back().sample_errors[sample_index];
  }
}

void RafkoDatasetCost::set_features_for_labels(
  const std::vector<std::vector<sdouble32>>& neuron_data,
  uint32 neuron_buffer_start_index, uint32 raw_start_index, uint32 labels_to_evaluate
){
  assert((raw_start_index + labels_to_evaluate) <= error_state.back().sample_errors.size());
  cost_function->get_feature_errors(
    dataset.get_label_samples(), neuron_data, error_state.back().sample_errors,
    raw_start_index, raw_start_index, labels_to_evaluate, neuron_buffer_start_index, dataset.get_number_of_label_samples()
  );
  if(!exposed_to_multithreading){
    error_state.back().error_sum = 0;
    error_calculation_threads.start_and_block(error_calculation_lambda);
  }
}

void RafkoDatasetCost::set_features_for_sequences(
  const std::vector<std::vector<sdouble32>>& neuron_data, uint32 neuron_buffer_start_index,
  uint32 sequence_start_index, uint32 sequences_to_evaluate,
  uint32 start_index_in_sequence, uint32 sequence_truncation,
  std::vector<sdouble32>& tmp_data
){
  if((sequence_start_index + sequences_to_evaluate) <= dataset.get_number_of_sequences()){
    if((start_index_in_sequence + sequence_truncation) > dataset.get_sequence_size())
      throw std::runtime_error("Sequence truncation indices out of bounds!");

    uint32 raw_start_index = sequence_start_index * dataset.get_sequence_size();
    uint32 labels_to_evaluate = sequences_to_evaluate * dataset.get_sequence_size();
    tmp_data.resize(labels_to_evaluate);

    cost_function->get_feature_errors(
      dataset.get_label_samples(), neuron_data, tmp_data,
      raw_start_index, 0, labels_to_evaluate, neuron_buffer_start_index, dataset.get_number_of_label_samples()
    );
    uint32 copy_index_base = 0;
    for(uint32 sequence_iterator = 0; sequence_iterator < sequences_to_evaluate; ++sequence_iterator){
      std::copy(
        tmp_data.begin() + copy_index_base + start_index_in_sequence,
        tmp_data.begin() + copy_index_base + start_index_in_sequence + sequence_truncation,
        error_state.back().sample_errors.begin() + raw_start_index + copy_index_base + start_index_in_sequence
      );
      copy_index_base += dataset.get_sequence_size();
    }

    if(!exposed_to_multithreading){
      error_state.back().error_sum = 0;
      error_calculation_threads.start_and_block(error_calculation_lambda);
    }
  }else throw std::runtime_error("Sequence index out of bounds!");
}

void RafkoDatasetCost::conceal_from_multithreading(){
  exposed_to_multithreading = false;
  error_state.back().error_sum = 0;
  error_calculation_threads.start_and_block(error_calculation_lambda);
}

void RafkoDatasetCost::accumulate_error_sum(uint32 error_start, uint32 errors_to_sum){
  sdouble32 local_error = 0;
  for(uint32 sample_index = error_start; sample_index < (error_start + errors_to_sum) ; ++sample_index)
    local_error += error_state.back().sample_errors[sample_index];
  std::lock_guard<std::mutex> my_lock(dataset_mutex);
  error_state.back().error_sum += local_error;
}

} /* namespace rafko_gym */

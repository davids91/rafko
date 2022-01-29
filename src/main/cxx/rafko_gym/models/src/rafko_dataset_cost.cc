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
#include <mutex>
#include <assert.h>
#include <functional>

namespace rafko_gym{

rafko_utilities::DataPool<sdouble32> RafkoDatasetCost::common_datapool(1,1);

sdouble32 RafkoDatasetCost::set_feature_for_label(const rafko_gym::RafkoEnvironment& environment, uint32 sample_index, const std::vector<sdouble32>& neuron_data){
  assert(environment.get_number_of_label_samples() > sample_index);
  return cost_function->get_feature_error(
    environment.get_label_sample(sample_index), neuron_data,
    environment.get_number_of_label_samples()
  );
}

sdouble32 RafkoDatasetCost::set_features_for_labels(
  const rafko_gym::RafkoEnvironment& environment, const std::vector<std::vector<sdouble32>>& neuron_data,
  uint32 neuron_buffer_start_index, uint32 raw_start_index, uint32 labels_to_evaluate
){
  assert((raw_start_index + labels_to_evaluate) <= environment.get_number_of_label_samples());

  std::vector<sdouble32>& error_labels = common_datapool.reserve_buffer(labels_to_evaluate);
  /*!Note: No need to fill the reserved buffer with initial values because every element of it will be overwritten */
  cost_function->get_feature_errors(
    environment.get_label_samples(), neuron_data, error_labels,
    raw_start_index, 0/* error start */, labels_to_evaluate, neuron_buffer_start_index,
    environment.get_number_of_label_samples()
  );

  sdouble32 error_sum = double_literal(0.0);
  std::mutex sum_mutex;
  error_calculation_threads.start_and_block( std::bind( &RafkoDatasetCost::accumulate_error_sum, this,
    std::ref(error_labels), std::ref(error_sum), labels_to_evaluate,
    std::ref(sum_mutex), std::placeholders::_1
  ) );
  common_datapool.release_buffer(error_labels);
  return error_sum;
}

sdouble32 RafkoDatasetCost::set_features_for_sequences(
  const rafko_gym::RafkoEnvironment& environment, const std::vector<std::vector<sdouble32>>& neuron_data,
  uint32 neuron_buffer_index, uint32 sequence_start_index, uint32 sequences_to_evaluate,
  uint32 start_index_in_sequence, uint32 sequence_truncation
){
  std::vector<sdouble32>& resulting_errors = common_datapool.reserve_buffer(sequences_to_evaluate * environment.get_sequence_size());
  sdouble32 error_sum = set_features_for_sequences(
    environment, neuron_data,
    neuron_buffer_index, sequence_start_index, sequences_to_evaluate,
    start_index_in_sequence, sequence_truncation, resulting_errors
  );
  common_datapool.release_buffer(resulting_errors);
  return error_sum;
}

sdouble32 RafkoDatasetCost::set_features_for_sequences(
  const rafko_gym::RafkoEnvironment& environment, const std::vector<std::vector<sdouble32>>& neuron_data,
  uint32 neuron_buffer_start_index, uint32 sequence_start_index, uint32 sequences_to_evaluate,
  uint32 start_index_in_sequence, uint32 sequence_truncation, std::vector<sdouble32>& tmp_data
){
  assert(environment.get_number_of_sequences() >= (sequence_start_index + sequences_to_evaluate));
  assert(environment.get_sequence_size() >= (start_index_in_sequence + sequence_truncation));

  uint32 raw_start_index = sequence_start_index * environment.get_sequence_size();
  uint32 labels_to_evaluate = sequences_to_evaluate * environment.get_sequence_size();
  std::vector<sdouble32>& error_labels = common_datapool.reserve_buffer(labels_to_evaluate);

  tmp_data.resize(labels_to_evaluate);
  std::fill(tmp_data.begin(), tmp_data.end(), double_literal(0.0));
  std::fill(error_labels.begin(), error_labels.end(), double_literal(0.0));

  cost_function->get_feature_errors(
    environment.get_label_samples(), neuron_data, tmp_data,
    raw_start_index, 0/* error_start */, labels_to_evaluate, neuron_buffer_start_index,
    environment.get_number_of_label_samples()
  );

  uint32 copy_index_base = 0;
  for(uint32 sequence_iterator = 0; sequence_iterator < sequences_to_evaluate; ++sequence_iterator){
    std::copy(
      tmp_data.begin() + copy_index_base + start_index_in_sequence,
      tmp_data.begin() + copy_index_base + start_index_in_sequence + sequence_truncation,
      error_labels.begin() + copy_index_base + start_index_in_sequence
    );
    copy_index_base += environment.get_sequence_size();
  }

  sdouble32 error_sum = double_literal(0.0);
  std::mutex sum_mutex;
  error_calculation_threads.start_and_block( std::bind( &RafkoDatasetCost::accumulate_error_sum, this,
    std::ref(error_labels), std::ref(error_sum), labels_to_evaluate,
    std::ref(sum_mutex), std::placeholders::_1
  ) );
  common_datapool.release_buffer(error_labels);
  return error_sum;
}

void RafkoDatasetCost::accumulate_error_sum(std::vector<sdouble32>& source, sdouble32& target, uint32 length, std::mutex& error_mutex, uint32 thread_index){
  sdouble32 local_error = double_literal(0.0);
  uint32 errors_to_sum_in_one_thread = (length / settings.get_sqrt_of_solve_threads()) + 1;
  uint32 error_start = errors_to_sum_in_one_thread * thread_index;
  errors_to_sum_in_one_thread = std::min(errors_to_sum_in_one_thread, (length - error_start));
  errors_to_sum_in_one_thread = std::min(errors_to_sum_in_one_thread, static_cast<uint32>(length - error_start));

  for(uint32 sample_index = error_start; sample_index < (error_start + errors_to_sum_in_one_thread) ; ++sample_index)
    local_error += source[sample_index];

  std::lock_guard<std::mutex> my_lock(error_mutex);
  target += local_error;
}

} /* namespace rafko_gym */

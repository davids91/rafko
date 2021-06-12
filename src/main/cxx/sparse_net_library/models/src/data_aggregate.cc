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

#include <cmath>

namespace sparse_net_library {

DataPool<sdouble32> Data_aggregate::common_datapool(1,1);

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
  const deque<vector<sdouble32>>& neuron_data,
  uint32 neuron_buffer_start_index, uint32 raw_start_index, uint32 labels_to_evaluate
){
  if((raw_start_index + labels_to_evaluate) <= error_state.back().sample_errors.size()){
    cost_function->get_feature_errors(
      label_samples, neuron_data, error_state.back().sample_errors,
      raw_start_index, raw_start_index, labels_to_evaluate, neuron_buffer_start_index, get_number_of_label_samples()
    );
    error_state.back().error_sum = 0;
    for(uint32 sample_index = 0; sample_index < label_samples.size() ; ++sample_index)
      error_state.back().error_sum += error_state.back().sample_errors[sample_index];
  }else throw new std::runtime_error("Label index out of bounds!");
}

void Data_aggregate::set_features_for_sequences(
  const deque<vector<sdouble32>>& neuron_data,
  uint32 neuron_buffer_start_index, uint32 raw_start_index, uint32 labels_to_evaluate,
  uint32 start_index_in_sequence, uint32 sequence_truncation
){
  if((raw_start_index + labels_to_evaluate) <= error_state.back().sample_errors.size()){
    vector<sdouble32>& resulting_errors = common_datapool.reserve_buffer(labels_to_evaluate);
    cost_function->get_feature_errors(
      label_samples, neuron_data, resulting_errors,
      raw_start_index, 0, labels_to_evaluate, neuron_buffer_start_index, get_number_of_label_samples()
    );
    std::copy(
      resulting_errors.begin(),resulting_errors.end(),
      error_state.back().sample_errors.begin() + raw_start_index
    );
    common_datapool.release_buffer(resulting_errors);
    error_state.back().error_sum = 0;
    for(uint32 sample_index = 0; sample_index < label_samples.size() ; ++sample_index)
      error_state.back().error_sum += error_state.back().sample_errors[sample_index];
  }else throw new std::runtime_error("Label index out of bounds!");
}

} /* namespace sparse_net_library */

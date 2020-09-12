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
#include "sparse_net_library/models/cost_function.h"

#include <cmath>

namespace sparse_net_library {

using std::min;
using std::ref;
using std::async;

void Cost_function::get_feature_errors(
  const vector<vector<sdouble32>>& labels, const vector<vector<sdouble32>>& neuron_data,
  vector<sdouble32>& errors_for_labels, uint32 label_start, uint32 sample_number
){
  if((label_start + neuron_data.size()) > labels.size())
    throw new std::runtime_error("Label index out of bounds with Neuron data!");

  uint32 neuron_data_start_index = 0;
  const uint32 labels_to_do_in_a_thread = 1 + static_cast<uint32>(neuron_data.size()/context.get_sqrt_of_process_threads());
  for(
    uint32 thread_index = 0;
    ((thread_index < context.get_sqrt_of_process_threads()) 
      && (neuron_data.size() > neuron_data_start_index));
    ++thread_index
  ){
    thread_results.push_back(vector<future<sdouble32>>());
    thread_results.back().reserve(context.get_sqrt_of_process_threads());
    process_threads.push_back(thread(
      &Cost_function::feature_errors_thread, this,
      ref(labels), ref(neuron_data), ref(errors_for_labels), label_start, neuron_data_start_index,
      min(labels_to_do_in_a_thread, static_cast<uint32>(neuron_data.size() - neuron_data_start_index)),
      thread_index, sample_number
    ));
    neuron_data_start_index += labels_to_do_in_a_thread;
    label_start += labels_to_do_in_a_thread;
  }
  while(0 < process_threads.size()){ /* wait for threads to finish! */
    if(process_threads.back().joinable()){
      process_threads.back().join();
      thread_results.pop_back();
      process_threads.pop_back();
    }
  }
}

void Cost_function::feature_errors_thread(
  const vector<vector<sdouble32>>& labels, const vector<vector<sdouble32>>& neuron_data, vector<sdouble32>& errors_for_labels,
  uint32 label_start, uint32 neuron_data_start_index, uint32 labels_to_process, uint32 outer_thread_index, uint32 sample_number
){
  for(uint32 label_iterator = 0; label_iterator < labels_to_process; ++label_iterator){
    errors_for_labels[label_start + label_iterator] = get_feature_error(
      labels[label_start + label_iterator], neuron_data[neuron_data_start_index + label_iterator],
      context.get_sqrt_of_process_threads(), outer_thread_index, sample_number
    );
  }
}

sdouble32 Cost_function::get_feature_error(const vector<sdouble32>& labels, const vector<sdouble32>& neuron_data, uint32 max_threads, uint32 outer_thread_index, uint32 sample_number){
  uint32 feature_start_index = neuron_data.size() - feature_size;
  sdouble32 error_value = 0;
  const uint32 feature_number = 1 + static_cast<uint32>(labels.size()/max_threads);
  for(uint32 thread_index = 0; ((thread_index < max_threads) && (neuron_data.size() > feature_start_index)); ++thread_index){
    thread_results[outer_thread_index].push_back(async(std::launch::async,
      &Cost_function::summarize_errors, this, ref(labels), ref(neuron_data), feature_start_index,
      min(feature_number, static_cast<uint32>(labels.size() - feature_start_index))
    ));
    feature_start_index += feature_number;
  }
  while(0 < thread_results[outer_thread_index].size()){ /* wait for threads */
    error_value += thread_results[outer_thread_index].back().get();
    thread_results[outer_thread_index].pop_back();
  }
  return error_post_process(error_value, sample_number);
}

sdouble32 Cost_function::summarize_errors(
  const vector<sdouble32>& labels, const vector<sdouble32>& neuron_data,
  uint32 neuron_data_start_index, uint32 number_to_add
){
  sdouble32 local_error = 0;
  for(uint32 feature_iterator = 0; feature_iterator < number_to_add; ++feature_iterator){
    local_error += get_cell_error( /* (start + feature - netsize) gives back the index in the label */
      labels[neuron_data_start_index + feature_iterator + feature_size - neuron_data.size()],
      neuron_data[neuron_data_start_index + feature_iterator]
    );
  }
  return local_error;
}

} /* namespace sparse_net_library */

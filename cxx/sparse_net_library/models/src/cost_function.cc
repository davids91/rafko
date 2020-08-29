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

using std::ref
using std::async;

sdouble32 Cost_function::get_feature_error(const vector<sdouble32>& labels, const vector<sdouble32>& neuron_data, uint32 sample_number){
  uint32 feature_start_index = neuron_data.size() - feature_size;
  sdouble32 error_value = 0;
  const uint32 feature_number = 1 + static_cast<uint32>(labels.size()/context.get_max_processing_threads());
  for(
    uint32 thread_index = 0;
    ( (thread_index < context.get_max_processing_threads()) && (neuron_data.size() > feature_start_index) );
    ++thread_index
  ){ /* For every provided sample */
    thread_results.push_back(async( std::launch::async,
      &Cost_function::summarize_errors, this,
      ref(labels), ref(neuron_data), feature_start_index,
      std::min(feature_number, static_cast<uint32>(neuron_data.size() - feature_start_index))
    ));
    feature_start_index += feature_number;
  }
  while(0 < thread_results.size()){ /* wait for threads */
    error_value += thread_results.back().get();
    thread_results.pop_back();
  }
  return error_post_process(error_value, sample_number);
}

sdouble32 Cost_function::summarize_errors(const vector<sdouble32>& labels, const vector<sdouble32>& neuron_data, uint32 start_index, uint32 number_to_add){
  sdouble32 local_error = 0;
  for(uint32 feature_iterator = 0; feature_iterator < number_to_add; ++feature_iterator){
    local_error += get_cell_error( /* (start + feature - netsize) gives back the index in the label */
      labels[start_index + feature_iterator + feature_size - neuron_data.size()],
      neuron_data[start_index + feature_iterator]
    );
  }
  return local_error;
}

} /* namespace sparse_net_library */

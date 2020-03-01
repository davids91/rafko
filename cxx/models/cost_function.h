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

#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H

#include "sparse_net_global.h"
#include "models/service_context.h"

#include <vector>
#include <thread>
#include <mutex>
#include <future>
#include <functional>
#include <cmath>

namespace sparse_net_library{

using std::vector;
using std::atomic;
using std::mutex;
using std::lock_guard;
using std::thread;

/**
 * @brief      Error function handling and utilities, provides a hook for a computation
 *             function to be run on every sample by feature.
 */
class Cost_function{
public:
  Cost_function(uint32 feature_size_, Service_context service_context = Service_context())
  :  context(service_context)
  ,  process_threads(0)
  ,  feature_size(feature_size_)
  ,  error_value(0)
  { process_threads.reserve(service_context.get_max_processing_threads()); };

  /**
   * @brief      Gets the error for a feature for a label set under the given index
   *
   * @param[in]  sample_index   The index of the sample in the dataset
   * @param[in]  label_index    The index of the datapoint inside the sample
   *
   * @return     The error.
   */
  sdouble32 get_error(const vector<sdouble32>& labels, const vector<sdouble32>& neuron_data){
    lock_guard<mutex> error_lock(error_mutex);
    error_value.store(0);
    uint32 feature_start_index = neuron_data.size() - feature_size;
    const uint32 feature_number = 1 + static_cast<uint32>(labels.size()/context.get_max_processing_threads());
    for(
      uint32 thread_index = 0;
      ( (thread_index < context.get_max_processing_threads())
        &&(labels.size() > feature_start_index) );
      ++thread_index
    ){ /* For every provided sample */
        process_threads.push_back(thread(
          &Cost_function::summarize_errors, this,
          ref(labels), ref(neuron_data), feature_start_index, 
          std::min(feature_number, static_cast<uint32>(labels.size() - feature_start_index))
        ));
        feature_start_index += feature_number;
    }
    wait_for_threads(process_threads);
    return error_value;
  }

  /**
   * @brief      Gets the the Cost function function derivative for a feature compared to a selected label set
   *
   * @param      feature_index  feature to examine
   * @param[in]  label_value    The value of the datapoint to compare
   * @param[in]  feature_value  The value to compare to label_value
   *
   * @return     The gradient of the cost function in regards to its input
   */
  sdouble32 get_d_cost_over_d_feature(uint32 feature_index, const vector<sdouble32>& label, const vector<sdouble32>& neuron_data) const{
    return get_d_cost_over_d_feature(label[feature_index],neuron_data[feature_index]);
  }

  virtual sdouble32 get_error(sdouble32 label_value, sdouble32 feature_value) const = 0;
  virtual sdouble32 get_d_cost_over_d_feature(sdouble32 label_value, sdouble32 feature_value) const = 0;

protected:
  Service_context context;
  vector<thread> process_threads;
  uint32 feature_size;
  mutex error_mutex;
  atomic<sdouble32> error_value;

  void summarize_errors(const vector<sdouble32>& labels, const vector<sdouble32>& neuron_data, uint32 start_index, uint32 number_to_add){
    sdouble32 buffer = error_value;
    sdouble32 local_error = 0;
    for(uint32 feature_iterator = 0; feature_iterator < number_to_add; ++feature_iterator){
      local_error += get_error(labels[start_index + feature_iterator],neuron_data[start_index + feature_iterator]);
    }
    while(!error_value.compare_exchange_weak(buffer,(buffer + local_error)))buffer = error_value;
  }

  /**
   * @brief      This function waits for the given threads to finish, ensures that every thread
   *             in the reference vector is finished, before it does.
   *
   * @param      calculate_threads  The calculate threads
   *//*!TODO: Find a better solution for these snippets */
  static void wait_for_threads(vector<thread>& threads){
    while(0 < threads.size()){
      if(threads.back().joinable()){
        threads.back().join();
        threads.pop_back();
      }
    }
  }
};

} /* namespace sparse_net_library */
#endif /* COST_FUNCTION_H */

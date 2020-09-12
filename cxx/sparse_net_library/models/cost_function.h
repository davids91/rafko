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

#include <vector>
#include <thread>
#include <future>

#include "rafko_mainframe/models/service_context.h"

namespace sparse_net_library{

using std::vector;
using std::thread;
using std::future;

using rafko_mainframe::Service_context;

/**
 * @brief      Error function handling and utilities, provides a hook for a computation
 *             function to be run on every sample by feature.
 */
class Cost_function{
public:
  Cost_function(uint32 feature_size_, cost_functions the_function_, Service_context& service_context)
  : context(service_context)
  , process_threads()
  , thread_results()
  , feature_size(feature_size_)
  , the_function(the_function_)
  {
    process_threads.reserve(context.get_sqrt_of_process_threads());
    thread_results.reserve(context.get_max_processing_threads());
  };

  /**
   * @brief      Gets the error for a feature for a label set under the given index
   *
   * @param[in]  labels         The array containing the labels to compare the given neuron data to
   * @param[in]  neuron_data    The neuron data to compare for the given labels array
   * @param[in]  sample_number  The overall count of the samples to be used in the final calculations(e.g. in mean squared error)
   *
   * @return     The feature error.
   */
  sdouble32 get_feature_error(const vector<sdouble32>& labels, const vector<sdouble32>& neuron_data, uint32 sample_number){
    thread_results.push_back(vector<future<sdouble32>>());
    thread_results.back().reserve(context.get_max_processing_threads());
    sdouble32 result_error = get_feature_error(labels, neuron_data, context.get_max_processing_threads(), 0, sample_number);
    thread_results.clear();
    return result_error;
  }

  /**
   * @brief      Gets the error produced by the sequences of the given label-data pair
   *
   * @param[in]  labels              The array containing the labels to compare the given neuron data to
   * @param[in]  neuron_data         The neuron data to compare for the given labels array
   * @param      errors_for_labels   The vector to load the resulting errors in, shall be of equal size to @labels
   * @param[in]  label_start         The index of the label to start evaluating the pairs from
   * @param[in]  labels_to_evaluate  The number of labels to evaluate
   * @param[in]  neuron_start        The starting index of the neuron data outer buffer
   * @param[in]  sample_number       The number of overall samples, required for post-processing
   */
  void get_feature_errors(
    const vector<vector<sdouble32>>& labels, const vector<vector<sdouble32>>& neuron_data, vector<sdouble32>& errors_for_labels,
    uint32 label_start, uint32 labels_to_evaluate, uint32 neuron_start, uint32 sample_number
  );

  /**
   * @brief      Gets the the Cost function function derivative for a feature compared to a selected label set
   *
   * @param      feature_index  feature to examine
   * @param[in]  label_value    The value of the datapoint to compare
   * @param[in]  feature_value  The value to compare to label_value
   *
   * @return     The gradient of the cost function in regards to its input
   */
  sdouble32 get_d_cost_over_d_feature(uint32 feature_index, const vector<sdouble32>& label, const vector<sdouble32>& neuron_data, uint32 sample_number) const{
    return error_post_process(get_d_cost_over_d_feature(
      label[feature_index], neuron_data[neuron_data.size() - feature_size + feature_index], sample_number
    ), sample_number);
  }

  /**
   * @brief      Gets the type of the implemented cost function.
   *
   * @return     The type.
   */
  cost_functions get_type(void){
    return the_function;
  }

  virtual ~Cost_function(void) = default;

protected:
  Service_context& context;
  vector<thread> process_threads;
  vector<vector<future<sdouble32>>> thread_results;
  uint32 feature_size;

  /**
   * @brief      The post-processing function to be provided by the implementer
   *
   * @param[in]  error_value    The raw error value
   * @param[in]  sample_number  The number of overall samples to be used in the relevant dataset
   *
   * @return     the final error value
   */
  virtual sdouble32 error_post_process(sdouble32 error_value, uint32 sample_number) const = 0;

  /**
   * @brief      Calculates the error for one number-pair inside the label-data pair
   *
   * @param[in]  label_value    The label value
   * @param[in]  feature_value  The data to comapre to the label value
   *
   * @return     The distance between the two given arguments
   */
  virtual sdouble32 get_cell_error(sdouble32 label_value, sdouble32 feature_value) const = 0;

  /**
   * @brief      The derivative function to be provided by the implementer
   *
   * @param[in]  label_value    The label value
   * @param[in]  feature_value  The data to comapre to the label value
   * @param[in]  sample_number  The number of overall samples to be used in the relevant dataset
   *
   * @return     The derivative of the elements of the label-data pair
   */
  virtual sdouble32 get_d_cost_over_d_feature(sdouble32 label_value, sdouble32 feature_value, uint32 sample_number) const = 0;

  /**
   * @brief      Corresponding to the other function in this class with the same name, without default one-threaded parameters given.
   *
   * @param[in]  labels              The array containing the labels to compare the given neuron data to
   * @param[in]  neuron_data         The neuron data to compare for the given labels array
   * @param[in]  max_threads         The maximum threads to be used for the curren processing run
   * @param[in]  outer_thread_index  The index to be used for this processing run, basically to find out which @thread_results array to use
   * @param[in]  sample_number       The overall count of the samples to be used in the final calculations(e.g. in mean squared error)
   *
   * @return     The overall error produced by the given label-data pair.
   */
  sdouble32 get_feature_error(const vector<sdouble32>& labels, const vector<sdouble32>& neuron_data, uint32 max_threads, uint32 outer_thread_index, uint32 sample_number);


  /**
   * @brief      Summarizes the errors given back by @get_cell_error for all of the features. It's called
   *             by @get_feature_error, which divides the features to almost equal parts,
   *             and calls this function on them.
   *
   * @param[in]  labels                   The labels
   * @param[in]  neuron_data              The neuron data
   * @param[in]  neuron_data_start_index  The start index of in the neuron data
   * @param[in]  number_to_add            The number of features to calculate
   *
   * @return     returns with the error summary under the range {start_index;(start_index + number_to_add)}
   */
  sdouble32 summarize_errors(
    const vector<sdouble32>& labels, const vector<sdouble32>& neuron_data,
    uint32 neuron_data_start_index, uint32 number_to_add
  );
private:
  cost_functions the_function; /* cost function type */

  /**
   * @brief      A Thread being used to sum up the error for each label-data pair and load the result into the provided error vector
   *
   * @param[in]  labels                   The label arrays to compare the data to
   * @param[in]  neuron_data              The neuron data to compare to the labels
   * @param      errors_for_labels        The vector to load the resulting errors in, shall be of equal size to @labels
   * @param[in]  label_start              The index of the label to start evaluating the pairs from
   * @param[in]  neuron_data_start_index  The index inside the neuron data corresponding to the start index defined for @labels
   * @param[in]  labels_to_process        The number of label-data pairs to process this thread
   * @param[in]  outer_thread_index       The outer thread index the index of the thread, to find out which @thread_results array to use
   * @param[in]  sample_number            The number of overall samples, required for post-processing
   */
  void feature_errors_thread(
    const vector<vector<sdouble32>>& labels, const vector<vector<sdouble32>>& neuron_data, vector<sdouble32>& errors_for_labels,
    uint32 label_start, uint32 neuron_data_start_index, uint32 labels_to_process, uint32 outer_thread_index, uint32 sample_number
  );
};

} /* namespace sparse_net_library */
#endif /* COST_FUNCTION_H */

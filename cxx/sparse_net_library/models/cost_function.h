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
#include <future>

#include "rafko_mainframe/models/service_context.h"

namespace sparse_net_library{

using std::vector;
using std::atomic;
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
  , thread_results()
  , feature_size(feature_size_)
  , the_function(the_function_)
  { thread_results.reserve(service_context.get_max_processing_threads()); };

  /**
   * @brief      Gets the error for a feature for a label set under the given index
   *
   * @param[in]  labels         The array containing the labels to compare the given neuron data to
   * @param[in]  neuron_data    The neuron data to compare for the given labels array
   * @param[in]  sample_number  The overall count of the samples to be used in the final calculations(e.g. in mean squared error)
   * @param[in]  thread_index   The index to be used for this processing run
   *
   * @return     The feature error.
   */
  sdouble32 get_feature_error(const vector<sdouble32>& labels, const vector<sdouble32>& neuron_data, uint32 sample_number);

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
  vector<future<sdouble32>> thread_results;
  uint32 feature_size;

  virtual sdouble32 error_post_process(sdouble32 error_value, uint32 sample_number) const = 0;
  virtual sdouble32 get_cell_error(sdouble32 label_value, sdouble32 feature_value) const = 0;
  virtual sdouble32 get_d_cost_over_d_feature(sdouble32 label_value, sdouble32 feature_value, uint32 sample_number) const = 0;

  /**
   * @brief      Summarizes the errors given back by @get_cell_error for all of the features. It's called
   *             by @get_feature_error, which divides the features to almost equal parts,
   *             and calls this function on them.
   *
   * @param[in]  labels         The labels
   * @param[in]  neuron_data    The neuron data
   * @param[in]  start_index    The start index of in the neuron data
   * @param[in]  number_to_add  The number of features to calculate
   *
   * @return     returns with the error summary under the range {start_index;(start_index + number_to_add)}
   */
  sdouble32 summarize_errors(const vector<sdouble32>& labels, const vector<sdouble32>& neuron_data, uint32 start_index, uint32 number_to_add);

private:
  cost_functions the_function; /* cost function type */
};

} /* namespace sparse_net_library */
#endif /* COST_FUNCTION_H */

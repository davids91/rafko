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

namespace sparse_net_library{

using std::vector;
using std::atomic;
using std::mutex;
using std::future;
using std::lock_guard;
using std::function;

/**
 * @brief      Error function handling and utilities, provides a hook for a computation
 *             function to be run on every sample by feature.
 */
class Cost_function{
public:
  Cost_function(vector<vector<sdouble32>>& label_samples, Service_context context = Service_context())
  : max_threads(context.get_max_processing_threads())
  , labels(label_samples)
  { };

  /**
   * @brief      Gets the overall error compared to the label set
   *
   * @return     The error.
   */
  virtual sdouble32 get_error(vector<vector<sdouble32>>& features) const = 0;

  /**
   * @brief      Gets the overall error comapred to a sample in the labelset
   *
   * @return     The error.
   */
  virtual sdouble32 get_error(uint32 sample_index, vector<sdouble32>& features) const = 0;

  /**
   * @brief      Gets the error for a feature for a label set under the given index
   *
   * @param[in]  sample_index   The index of the sample in the dataset
   * @param[in]  label_index    The index of the datapoint inside the sample
   * @param[in]  feature_value  The value of the datapoint to compare to
   *
   * @return     The error.
   */
  virtual sdouble32 get_error(uint32 sample_index, uint32 label_index, sdouble32 feature_value) const = 0;

  /**
   * @brief      Gets the the Cost function function derivative for a feature compared to a selected label set
   *
   * @param[in]  sample_index   The index of the sample in the dataset
   * @param[in]  label_index    The index of the datapoint inside the sample
   * @param[in]  feature_value  The value of the datapoint to compare to
   *
   * @return     The d cost over d feature.
   */
  virtual sdouble32 get_d_cost_over_d_feature(uint32 sample_index, uint32 label_index, sdouble32 feature_value) const = 0;

protected:
  uint8 max_threads;
  vector<vector<sdouble32>>& labels;

  /**
   * @brief      Throws an exception if the references for a given features are incorrectly set up
   *
   * @param[in]  feature_index  The feature index
   */
  void verify_sizes(uint32 label_index, vector<sdouble32>& features) const{
    if(/* Check if input index is valid */
      (labels.size() <= label_index)
      ||(features.size() != labels[label_index].size())
    )throw "Incompatible Feature and Label sizes!";
  }
};

} /* namespace sparse_net_library */
#endif /* COST_FUNCTION_H */

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
 *    along with Foobar.  If not, see <https://www.gnu.org/licenses/> or
 *    <https://github.com/davids91/rafko/blob/master/LICENSE>
 */

#ifndef DATA_AGGREGATE_H
#define DATA_AGGREGATE_H

#include "sparse_net_global.h"
#include "gen/common.pb.h"
#include "models/cost_function.h"

#include <vector>
#include <limits>

namespace sparse_net_library{

using std::vector;

/**
 * @brief      A Data set container complete with adaptive error statistics
 */
class Data_aggregate{
public:
  Data_aggregate(Data_set& samples_, Cost_function& cost_function_)
  :  input_samples(samples_.feature_size())
  ,  label_samples(samples_.feature_size())
  ,  sample_errors(samples_.feature_size(),std::numeric_limits<sdouble32>::max())
  ,  average_error(std::numeric_limits<sdouble32>::max())
  ,  cost_function(cost_function_)
  { fill(samples_); }

  Data_aggregate(
    vector<vector<sdouble32>>&& input_samples_,
    vector<vector<sdouble32>>&& label_samples_,
    Cost_function& cost_function_
  ):  input_samples(input_samples_)
  ,  label_samples(label_samples_)
  ,  sample_errors(label_samples_.size())
  ,  average_error(std::numeric_limits<sdouble32>::max())
  ,  cost_function(cost_function_)
  { }

  void set_feature_for_label(uint32 sample_index, const vector<sdouble32>& neuron_data);

  const vector<sdouble32>& get_input_sample(uint32 sample_index){
    if(label_samples.size() > sample_index)
      return input_samples[sample_index];
      else throw "Sample index out of bounds!";
  }

  const vector<sdouble32>& get_label_sample(uint32 sample_index){
    if(label_samples.size() > sample_index)
      return label_samples[sample_index];
      else throw "Sample index out of bounds!";
  }

  sdouble32 get_error(void){
    return average_error;
  }

  uint32 get_feature_size(void){
    return label_samples[0].size();
  }

  uint32 get_number_of_samples(void){
    return label_samples.size();
  }

private:
  vector<vector<sdouble32>> input_samples;
  vector<vector<sdouble32>> label_samples;
  vector<sdouble32> sample_errors;
  atomic<sdouble32> average_error;
  Cost_function& cost_function;

  void fill(Data_set& samples);
};

} /* namespace sparse_net_library */

#endif /* DATA_AGGREGATE_H */

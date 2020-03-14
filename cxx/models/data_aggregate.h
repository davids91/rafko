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
#include "models/service_context.h"
#include "models/cost_function.h"
#include "services/function_factory.h"

#include <vector>
#include <memory>

namespace sparse_net_library{

using std::vector;
using std::unique_ptr;
using std::move;

/**
 * @brief      A Data set container complete with adaptive error statistics
 */
class Data_aggregate{
public:
  Data_aggregate(Data_set& samples_, unique_ptr<Cost_function> cost_function_)
  :  sample_number(static_cast<uint32>(samples_.labels_size()/samples_.feature_size()))
  ,  input_samples(sample_number)
  ,  label_samples(sample_number)
  ,  sample_errors(sample_number,1.0L)
  ,  error_sum(sample_number)
  ,  cost_function(move(cost_function_))
  { fill(samples_); }

  Data_aggregate(
    vector<vector<sdouble32>>&& input_samples_,
    vector<vector<sdouble32>>&& label_samples_,
    unique_ptr<Cost_function> cost_function_
  ):  sample_number(input_samples_.size())
  ,  input_samples(sample_number)
  ,  label_samples(sample_number)
  ,  sample_errors(sample_number,1.0L)
  ,  error_sum(sample_number)
  ,  cost_function(move(cost_function_))
  { }

  Data_aggregate(
    vector<vector<sdouble32>>&& input_samples_,
    vector<vector<sdouble32>>&& label_samples_,
    SparseNet& net, Service_context context = Service_context()
  ):  sample_number(input_samples_.size())
  ,  input_samples(input_samples_)
  ,  label_samples(label_samples_)
  ,  sample_errors(sample_number,1.0L)
  ,  error_sum(sample_number)
  ,  cost_function(Function_factory::build_cost_function(net, sample_number, context))
  { }

  void set_feature_for_label(uint32 sample_index, const vector<sdouble32>& neuron_data);

  void reset_errors(void){
    for(uint32 i = 0; i<get_number_of_samples(); ++i)
      sample_errors[i] = 1.0L;
    error_sum.store(get_number_of_samples());
  }

  const vector<sdouble32>& get_input_sample(uint32 sample_index){
    if(sample_number > sample_index)
      return input_samples[sample_index];
      else throw "Sample index out of bounds!";
  }

  const vector<sdouble32>& get_label_sample(uint32 sample_index){
    if(sample_number > sample_index)
      return label_samples[sample_index];
      else throw "Sample index out of bounds!";
  }

  sdouble32 get_error(uint32 index){
    if(sample_errors.size() > index)
      return sample_errors[index];
    else throw "Sample index out of bounds!";
  }

  sdouble32 get_error(void){
    return error_sum;
  }

  uint32 get_feature_size(void){
    return label_samples[0].size();
  }

  uint32 get_number_of_samples(void){
    return sample_number;
  }

private:
  uint32 sample_number;
  vector<vector<sdouble32>> input_samples;
  vector<vector<sdouble32>> label_samples;
  vector<sdouble32> sample_errors;
  atomic<sdouble32> error_sum;
  unique_ptr<Cost_function> cost_function;

  void fill(Data_set& samples);
};

} /* namespace sparse_net_library */

#endif /* DATA_AGGREGATE_H */

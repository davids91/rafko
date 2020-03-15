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
  ,  sample_errors(sample_number,(double_literal(1.0)/sample_number))
  ,  error_sum(double_literal(1.0))
  ,  cost_function(move(cost_function_))
  { fill(samples_); }

  Data_aggregate(
    vector<vector<sdouble32>>&& input_samples_,
    vector<vector<sdouble32>>&& label_samples_,
    unique_ptr<Cost_function> cost_function_
  ):  sample_number(input_samples_.size())
  ,  input_samples(sample_number)
  ,  label_samples(sample_number)
  ,  sample_errors(sample_number,(double_literal(1.0)/sample_number))
  ,  error_sum(double_literal(1.0))
  ,  cost_function(move(cost_function_))
  { }

  Data_aggregate(
    vector<vector<sdouble32>>&& input_samples_,
    vector<vector<sdouble32>>&& label_samples_,
    SparseNet& net, Service_context context = Service_context()
  ):  sample_number(input_samples_.size())
  ,  input_samples(input_samples_)
  ,  label_samples(label_samples_)
  ,  sample_errors(sample_number,(double_literal(1.0)/sample_number))
  ,  error_sum(double_literal(1.0))
  ,  cost_function(Function_factory::build_cost_function(net, sample_number, context))
  { }

  /**
   * @brief      Sets the approximated value for an observed value, 
   *             and updates the calculated errorbased on the cost function and the given value.
   *
   * @param[in]  sample_index  The sample index
   * @param[in]  neuron_data   The neuron data
   */
  void set_feature_for_label(uint32 sample_index, const vector<sdouble32>& neuron_data);

  /**
   * @brief      Sets the error values to the default value
   */
  void reset_errors(void){
    for(uint32 i = 0; i<get_number_of_samples(); ++i)
      sample_errors[i] = (double_literal(1.0)/sample_number);
    error_sum.store(double_literal(1.0));
  }

  /**
   * @brief      Gets an input sample from the set
   *
   * @param[in]  sample_index  The sample index
   *
   * @return     The input sample.
   */
  const vector<sdouble32>& get_input_sample(uint32 sample_index){
    if(sample_number > sample_index)
      return input_samples[sample_index];
      else throw "Sample index out of bounds!";
  }

  /**
   * @brief      Gets a label sample from the set
   *
   * @param[in]  sample_index  The sample index
   *
   * @return     The label sample.
   */
  const vector<sdouble32>& get_label_sample(uint32 sample_index){
    if(sample_number > sample_index)
      return label_samples[sample_index];
      else throw "Sample index out of bounds!";
  }

  /**
   * @brief      Returns the stored error for the sample under the index
   *
   * @param[in]  index  The index
   *
   * @return     The error.
   */
  sdouble32 get_error(uint32 index){
    if(sample_errors.size() > index)
      return sample_errors[index];
    else throw "Sample index out of bounds!";
  }

  /**
   * @brief      Gets the overall stored error.
   *
   * @return     The sum of the errors for all of the samples.
   */
  sdouble32 get_error(void){
    return error_sum;
  }

  /**
   * @brief      Gets the number of Neuron present in the output
   *
   * @return     The feature size.
   */
  uint32 get_feature_size(void){
    return label_samples[0].size();
  }

  /**
   * @brief      Gets the number of samples.
   *
   * @return     The number of samples.
   */
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

  /**
   * @brief      Converting the @Data_set message to vectors
   *
   * @param      samples  The data set to parse
   */
  void fill(Data_set& samples);
};

} /* namespace sparse_net_library */

#endif /* DATA_AGGREGATE_H */

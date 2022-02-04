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

#include "rafko_global.h"

#include <vector>
#include <thread>
#include <future>
#if(RAFKO_USES_OPENCL)
#include <utility>
#include <string>
#include <regex>
#include <CL/opencl.hpp>
#endif/*(RAFKO_USES_OPENCL)*/

#include "rafko_utilities/services/thread_group.h"
#include "rafko_mainframe/models/rafko_settings.h"
#if(RAFKO_USES_OPENCL)
#include "rafko_mainframe/models/rafko_nbuf_shape.h"
#include "rafko_mainframe/models/rafko_gpu_strategy_phase.h"
#endif/*(RAFKO_USES_OPENCL)*/

namespace rafko_gym{

/**
 * @brief      Error function handling and utilities, provides a hook for a computation
 *             function to be run on every sample by feature.
 */
class RAFKO_FULL_EXPORT CostFunction
#if(RAFKO_USES_OPENCL)
: public rafko_mainframe::RafkoGPUStrategyPhase
#endif/*(RAFKO_USES_OPENCL)*/
{
public:
  CostFunction(Cost_functions the_function_, rafko_mainframe::RafkoSettings& settings)
  : settings(settings)
  , the_function(the_function_)
  , execution_threads(settings.get_sqrt_of_solve_threads())
  {
    process_threads.reserve(settings.get_sqrt_of_solve_threads());
    for(uint32 thread_index = 0; thread_index < settings.get_max_solve_threads(); ++thread_index){
      thread_results.push_back(std::vector<std::future<sdouble32>>());
      thread_results.back().reserve(settings.get_sqrt_of_solve_threads());
    }
  };

  /**
   * @brief      Gets the error for a feature for a feature-label pair under the given index
   *
   * @param[in]  label          The array containing the label array to compare the given neuron data to
   * @param[in]  neuron_data    The neuron data to compare for the given label array
   * @param[in]  sample_number  The overall count of the samples to be used in the final calculations(e.g. in mean squared error)
   *
   * @return     The feature error.
   */
  sdouble32 get_feature_error(const std::vector<sdouble32>& label, const std::vector<sdouble32>& neuron_data, uint32 sample_number){
    return get_feature_error(label, neuron_data, settings.get_sqrt_of_solve_threads(), 0, sample_number);
  }

  /**
   * @brief      Corresponding to the other function in this class with the same name, without default one-threaded parameters given.
   *
   * @param[in]  label               The array containing the label to compare the given neuron data to
   * @param[in]  neuron_data         The neuron data to compare for the given label array
   * @param[in]  max_threads         The maximum threads to be used for the curren processing run
   * @param[in]  outer_thread_index  The index to be used for this processing run, basically to find out which @thread_results array to use
   * @param[in]  sample_number       The overall count of the samples to be used in the final calculations(e.g. in mean squared error)
   *
   * @return     The overall error produced by the given label-data pair.
   */
  sdouble32 get_feature_error(const std::vector<sdouble32>& label, const std::vector<sdouble32>& neuron_data, uint32 max_threads, uint32 outer_thread_index, uint32 sample_number);

  /**
   * @brief      Gets the error produced by the sequences of the given label-data pair
   *
   * @param[in]  labels              The array containing the labels to compare the given neuron data to
   * @param[in]  neuron_data         The neuron data to compare for the given labels array
   * @param      errors_for_labels   The vector to load the resulting errors in, size shall equal @labels_to_evaluate
   * @param[in]  label_start         The index of the label to start evaluating the pairs from
   * @param[in]  labels_to_evaluate  The number of labels to evaluate
   * @param[in]  neuron_start        The starting index of the neuron data outer buffer
   * @param[in]  neuron_start        The starting index inside the vector for the output errors @errors_for_labels
   * @param[in]  sample_number       The number of overall samples, required for post-processing
   */
  void get_feature_errors(
    const std::vector<std::vector<sdouble32>>& labels, const std::vector<std::vector<sdouble32>>& neuron_data, std::vector<sdouble32>& errors_for_labels,
    uint32 label_start, uint32 error_start, uint32 labels_to_evaluate, uint32 neuron_start, uint32 sample_number
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
  sdouble32 get_d_cost_over_d_feature(uint32 feature_index, const std::vector<sdouble32>& label, const std::vector<sdouble32>& neuron_data, uint32 sample_number) const{
    return error_post_process(get_d_cost_over_d_feature(
      label[feature_index], neuron_data[feature_index], sample_number
    ), sample_number);
  }

  /**
   * @brief      Gets the type of the implemented cost function.
   *
   * @return     The type.
   */
  Cost_functions get_type(){
    return the_function;
  }

  virtual ~CostFunction() = default;

  #if(RAFKO_USES_OPENCL)

  void set_parameters(uint32 pairs_to_evaluate_, uint32 feature_size_){
    pairs_to_evaluate = pairs_to_evaluate_;
    feature_size = feature_size_;
  }

  virtual std::string get_operation_kernel_source(std::string label_value, std::string feature_value) const = 0;
  virtual std::string get_post_process_kernel_source(std::string error_value) const = 0;

  cl::Program::Sources get_step_sources()const;
  std::vector<std::string> get_step_names()const;

  /**
   * @brief      Provides the input dimension of the cost function: a configured number of feature-label pairs to evaluate
   *
   * @return     Vector of dimensions in order of @get_step_sources and @get_step_names
   */
  std::vector<rafko_mainframe::RafkoNBufShape> get_input_shapes()const  {
    return { rafko_mainframe::RafkoNBufShape{ /* inputs and labels */
      pairs_to_evaluate * feature_size,
      pairs_to_evaluate * feature_size
    } };
  }

  /**
   * @brief      Provides the output dimension of the cost function: one error value for every number of pairs to evaluate
   *
   * @return     Vector of dimensions in order of @get_step_sources and @get_step_names
   */
  std::vector<rafko_mainframe::RafkoNBufShape> get_output_shapes()const  {
    return { rafko_mainframe::RafkoNBufShape{ 1u } };
  }
  std::tuple<cl::NDRange,cl::NDRange,cl::NDRange> get_solution_space()  {
    return std::make_tuple(cl::NullRange,cl::NDRange(pairs_to_evaluate),cl::NullRange);
  }

  #endif/*(RAFKO_USES_OPENCL)*/


protected:
  rafko_mainframe::RafkoSettings& settings;
  std::vector<std::thread> process_threads;
  std::vector<std::vector<std::future<sdouble32>>> thread_results;

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
   * @brief      Summarizes the errors given back by @get_cell_error for all of the features. It's called
   *             by @get_feature_error, which divides the features to almost equal parts,
   *             and calls this function on them.
   *
   * @param[in]  labels                 The labels
   * @param[in]  neuron_data            The neuron data
   * @param[in]  feature_start_index    The start index to start comparison from
   * @param[in]  number_to_eval         The number of features to calculate
   *
   * @return     returns with the error summary under the range {start_index;(start_index + number_to_add)}
   */
  sdouble32 summarize_errors(
    const std::vector<sdouble32>& labels, const std::vector<sdouble32>& neuron_data,
    uint32 feature_start_index, uint32 number_to_eval
  );
private:
  Cost_functions the_function; /* cost function type */
  rafko_utilities::ThreadGroup execution_threads;
  #if(RAFKO_USES_OPENCL)
  uint32 pairs_to_evaluate = 1u;
  uint32 feature_size = 1u;
  #endif/*(RAFKO_USES_OPENCL)*/


  /**
   * @brief      A Thread being used to sum up the error for each label-data pair and load the result into the provided error vector
   *
   * @param[in]  labels                                 The label arrays to compare the data to
   * @param[in]  neuron_data                            The neuron data to compare to the labels
   * @param      errors_for_labels                      The vector to load the resulting errors in, size shall equal @labels_to_evaluate
   * @param[in]  label_start                            The index of the label to start evaluating the pairs from
   * @param[in]  error_start                            The starting index in the label error vector this thread starts
   * @param[in]  neuron_data_start_index                The index inside the neuron data corresponding to the start index defined for @labels
   * @param[in]  labels_to_evaluate_in_one_thread       The maximum number of label-data pairs to process in one thread ( thread might process less, based on the size of @labels)
   * @param[in]  labels_evaluating_overall              The number of label-data pairs to process in the parent function call overall ( neuron data buffer array might not necessarily indicate the maximum size, as there might be other data cached next to it )
   * @param[in]  sample_number                          The number of overall samples, required for post-processing
   * @param[in]  thread_index                           The index of the thread the errors are accumulated in
   */
  void feature_errors_thread(
    const std::vector<std::vector<sdouble32>>& labels, const std::vector<std::vector<sdouble32>>& neuron_data, std::vector<sdouble32>& errors_for_labels,
    uint32 label_start, uint32 error_start, uint32 neuron_data_start_index,
    uint32 labels_to_evaluate_in_one_thread, uint32 labels_evaluating_overall, uint32 sample_number, uint32 thread_index
  );
};

} /* namespace rafko_gym */
#endif /* COST_FUNCTION_H */
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

#ifndef RAFKO_COST_FUNCTION_H
#define RAFKO_COST_FUNCTION_H

#include "rafko_global.hpp"

#include <vector>
#include <thread>
#include <future>
#if(RAFKO_USES_OPENCL)
#include <utility>
#include <string>
#include <regex>
#include <CL/opencl.hpp>
#endif/*(RAFKO_USES_OPENCL)*/

#include "rafko_utilities/services/thread_group.hpp"
#include "rafko_mainframe/models/rafko_settings.hpp"
#if(RAFKO_USES_OPENCL)
#include "rafko_mainframe/models/rafko_nbuf_shape.hpp"
#include "rafko_mainframe/models/rafko_gpu_strategy_phase.hpp"
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
  CostFunction(Cost_functions the_function, const rafko_mainframe::RafkoSettings& settings)
  : settings(settings)
  , m_theFunction(the_function)
  , m_executionThreads(settings.get_sqrt_of_solve_threads())
  {
    process_threads.reserve(settings.get_sqrt_of_solve_threads());
    for(std::uint32_t thread_index = 0; thread_index < settings.get_max_solve_threads(); ++thread_index){
      thread_results.push_back(std::vector<std::future<double>>());
      thread_results.back().reserve(settings.get_sqrt_of_solve_threads());
    }
  };

  /**
   * @brief      Gets the error for a feature-label pair under the given index
   *
   * @param[in]  label          The array containing the label array to compare the given neuron data to
   * @param[in]  neuron_data    The neuron data to compare for the given label array
   * @param[in]  sample_number  The overall count of the samples to be used in the final calculations(e.g. in mean squared error)
   *
   * @return     The feature error.
   */
  double get_feature_error(
    const std::vector<double>& label, const std::vector<double>& neuron_data,
    std::uint32_t sample_number
  ) const{
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
  double get_feature_error(
    const std::vector<double>& label, const std::vector<double>& neuron_data,
    std::uint32_t max_threads, std::uint32_t outer_thread_index, std::uint32_t sample_number
  ) const;

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
    const std::vector<std::vector<double>>& labels, const std::vector<std::vector<double>>& neuron_data, std::vector<double>& errors_for_labels,
    std::uint32_t label_start, std::uint32_t error_start, std::uint32_t labels_to_evaluate, std::uint32_t neuron_start, std::uint32_t sample_number
  ) const;

  /**
   * @brief      Gets the type of the implemented cost function.
   *
   * @return     The type.
   */
  constexpr Cost_functions get_type() const{
    return m_theFunction;
  }

  /**
   * @brief      Calculates the derivative for one number-pair inside the label-data pair
   *
   * @param[in]  label_value    The label value
   * @param[in]  feature_value  The data to comapre to the label value
   * @param[in]  feature_d      The derivative of the of the feature value
   *
   * @return     The distance between the two given arguments
   */
   virtual double get_derivative(
     double label_value, double feature_value,
     double feature_d, double sample_number
   ) const = 0;


  virtual ~CostFunction() = default;

  #if(RAFKO_USES_OPENCL)

  constexpr void set_parameters(std::uint32_t pairs_to_evaluate, std::uint32_t feature_size){
    m_pairsToEvaluate = pairs_to_evaluate;
    m_featureSize = feature_size;
  }

  /**
   * @brief   Provides the GPU kernel sources that implements part of the cost function
   *
   * @param[in]  label_value      The label value
   * @param[in]  feature_value    The data to comapre to the label value
   *
   * @return    The source implementing this part of the cost function in the GPU as Kernels
   */
  virtual std::string get_operation_kernel_source(std::string label_value, std::string feature_value) const = 0;

  /**
   * @brief   Provides the GPU kernel sources that implements part of the cost function
   *
   * @param[in]  error_value    The error value to post process
   *
   * @return    The source implementing this part of the cost function in the GPU as Kernels
   */
  virtual std::string get_post_process_kernel_source(std::string error_value) const = 0;

  /**
   * @brief      Provides the kernel function for the derivative of the objective
   *
   * @param[in]  label_value      The label value
   * @param[in]  feature_value    The data to comapre to the label value
   * @param[in]  feature_d        The derivative of the of the feature value
   * @param[in]  sample_number    The number of sample values the objective is evaluated on at once
   *
   * @return     The source for implementing the kernel of the derivative of the cost function
   */
  virtual std::string get_derivative_kernel_source(
    std::string label_value, std::string feature_value, std::string feature_d, std::string sample_number
  ) const = 0;

  /* +++ Functions taken from rafko_mainframe::RafkoGPUStrategyPhase */
  cl::Program::Sources get_step_sources() const override;
  std::vector<std::string> get_step_names() const override;
  std::vector<rafko_mainframe::RafkoNBufShape> get_input_shapes() const override{
    return { rafko_mainframe::RafkoNBufShape{ /* inputs and labels */
      m_pairsToEvaluate * m_featureSize,
      m_pairsToEvaluate * m_featureSize
    } };
  }
  std::vector<rafko_mainframe::RafkoNBufShape> get_output_shapes() const override{
    return { rafko_mainframe::RafkoNBufShape{ 1u } };
  }
  std::tuple<cl::NDRange,cl::NDRange,cl::NDRange> get_solution_space() const override{
    return std::make_tuple(cl::NullRange,cl::NDRange(m_pairsToEvaluate),cl::NullRange);
  }
  #endif/*(RAFKO_USES_OPENCL)*/


protected:
  const rafko_mainframe::RafkoSettings& settings;
  std::vector<std::thread> process_threads;
  mutable std::vector<std::vector<std::future<double>>> thread_results;

  /**
   * @brief      The post-processing function to be provided by the implementer
   *
   * @param[in]  error_value    The raw error value
   * @param[in]  sample_number  The number of overall samples to be used in the relevant dataset
   *
   * @return     the final error value
   */
  [[nodiscard]] virtual double error_post_process(double error_value, std::uint32_t sample_number) const = 0;

  /**
   * @brief      Calculates the error for one number-pair inside the label-data pair
   *
   * @param[in]  label_value    The label value
   * @param[in]  feature_value  The data to comapre to the label value
   *
   * @return     The distance between the two given arguments
   */
  virtual double get_cell_error(double label_value, double feature_value) const = 0;

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
  double summarize_errors(
    const std::vector<double>& labels, const std::vector<double>& neuron_data,
    std::uint32_t feature_start_index, std::uint32_t number_to_eval
  ) const;
private:
  Cost_functions m_theFunction; /* cost function type */
  rafko_utilities::ThreadGroup m_executionThreads;
  #if(RAFKO_USES_OPENCL)
  std::uint32_t m_pairsToEvaluate = 1u;
  std::uint32_t m_featureSize = 1u;
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
    const std::vector<std::vector<double>>& labels, const std::vector<std::vector<double>>& neuron_data, std::vector<double>& errors_for_labels,
    std::uint32_t label_start, std::uint32_t error_start, std::uint32_t neuron_data_start_index,
    std::uint32_t labels_to_evaluate_in_one_thread, std::uint32_t labels_evaluating_overall, std::uint32_t sample_number, std::uint32_t thread_index
  ) const;
};

} /* namespace rafko_gym */
#endif /* RAFKO_COST_FUNCTION_H */

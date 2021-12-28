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

#include "rafko_global.h"

#include <vector>
#include <memory>
#include <mutex>
#include <tuple>
#include <assert.h>

#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_utilities/services/thread_group.h"
#include "rafko_utilities/models/data_pool.h"
#include "rafko_net/models/cost_function.h"
#include "rafko_net/services/function_factory.h"

#include "rafko_gym/models/rafko_dataset_wrapper.h"
#include "rafko_gym/services/rafko_agent.h"

namespace rafko_gym{

/**
 * @brief      Error Statistics for a @RafkoDatasetWrapper
 */
class RAFKO_FULL_EXPORT DataAggregate{
public:
  DataAggregate(rafko_mainframe::RafkoSettings& settings_, const rafko_gym::RafkoDatasetWrapper& dataset_, std::shared_ptr<rafko_net::CostFunction> cost_function_)
  : settings(settings_)
  , dataset(dataset_)
  , error_state(double_literal(1.0),{
      std::vector<sdouble32>(dataset.get_number_of_label_samples(),(double_literal(1.0)/dataset.get_number_of_label_samples())), double_literal(1.0)
    })
  , cost_function(cost_function_)
  , error_calculation_threads(settings_.get_sqrt_of_solve_threads())
  { }

  DataAggregate(rafko_mainframe::RafkoSettings& settings_, const rafko_gym::RafkoDatasetWrapper& dataset_, rafko_net::Cost_functions the_function)
  : settings(settings_)
  , dataset(dataset_)
  , error_state(double_literal(1.0),{
      std::vector<sdouble32>(dataset.get_number_of_label_samples(),(double_literal(1.0)/dataset.get_number_of_label_samples())), double_literal(1.0)
    })
  , cost_function(rafko_net::FunctionFactory::build_cost_function(the_function, settings_))
  , exposed_to_multithreading(false)
  , error_calculation_threads(settings_.get_sqrt_of_solve_threads())
  { }

  /**
   * @brief      Sets the approximated value for an observed value,
   *             and updates the calculated errorbased on the cost function and the given value.
   *
   * @param[in]  sample_index  The sample index
   * @param[in]  neuron_data   The neuron data
   */
  void set_feature_for_label(uint32 sample_index, const std::vector<sdouble32>& neuron_data);

  /**
   * @brief      Same as @set_feature_for_label but in bulk
   *
   * @param[in]  neuron_data              The neuron data
   * @param[in]  neuron_buffer_index      The index of the outer neuron bufer to start evaluation from
   * @param[in]  raw_start_index          The raw start index inside the dataset labels; Meaning the index inside the labels array, which contains the samples(each with possible multiple labels in sequential order)
   * @param[in]  labels_to_evaluate       The labels to evaluate
   */
  void set_features_for_labels(
    const std::vector<std::vector<sdouble32>>& neuron_data,
    uint32 neuron_buffer_index, uint32 raw_start_index, uint32 labels_to_evaluate
  );

  /**
   * @brief      Same as @set_feature_for_label but in bulk
   *
   * @param[in]  neuron_data              The neuron data containing every output data for the @sequences_to_evaluate
   * @param[in]  neuron_buffer_index      The index of the outer neuron bufer to start evaluation from
   * @param[in]  sequence_start_index     The raw start index inside the dataset labels; Meaning the index inside the labels array, which contains the samples(each with possible multiple labels in sequential order)
   * @param[in]  sequences_to_evaluate    The labels to evaluate
   * @param[in]  start_index_in_sequence  The starting index inside each sequence to update the labels
   * @param[in]  sequence_truncation      The sequence truncation
   */
  void set_features_for_sequences(
    const std::vector<std::vector<sdouble32>>& neuron_data, uint32 neuron_buffer_index,
    uint32 sequence_start_index, uint32 sequences_to_evaluate,
    uint32 start_index_in_sequence, uint32 sequence_truncation
  ){
    std::vector<sdouble32>& resulting_errors = common_datapool.reserve_buffer(sequences_to_evaluate * dataset.get_sequence_size());
    set_features_for_sequences(
      neuron_data, neuron_buffer_index,
      sequence_start_index, sequences_to_evaluate, start_index_in_sequence, sequence_truncation,
      resulting_errors
    );
    common_datapool.release_buffer(resulting_errors);
  }

  /**
   * @brief      Same as @set_feature_for_label but in bulk
   *
   * @param[in]  neuron_data              The neuron data containing every output data for the @sequences_to_evaluate
   * @param[in]  neuron_buffer_index      The index of the outer neuron bufer to start evaluation from
   * @param[in]  sequence_start_index     The raw start index inside the dataset labels; Meaning the index inside the labels array, which contains the samples(each with possible multiple labels in sequential order)
   * @param[in]  sequences_to_evaluate    The labels to evaluate
   * @param[in]  start_index_in_sequence  The starting index inside each sequence to update the labels
   * @param[in]  sequence_truncation      The sequence truncation
   */
  void set_features_for_sequences(
    const std::vector<std::vector<sdouble32>>& neuron_data, uint32 neuron_buffer_index,
    uint32 sequence_start_index, uint32 sequences_to_evaluate,
    uint32 start_index_in_sequence, uint32 sequence_truncation,
    std::vector<sdouble32>& tmp_data
  );

  /**
   * @brief      Sets the error values to the default value
   */
  void reset_errors(){
    if(!exposed_to_multithreading){
      std::lock_guard<std::mutex> my_lock(dataset_mutex);
      for(sdouble32& sample_error : error_state.back().sample_errors)
        sample_error = (double_literal(1.0)/dataset.get_number_of_label_samples());
      error_state.back().error_sum = double_literal(1.0);
    }else throw std::runtime_error("Can't reset errors while set is exposed to multithreading!");
  }

  /**
   * @brief      Stores the current error values for later re-use
   */
  void push_state(){
    if(!exposed_to_multithreading){
      std::lock_guard<std::mutex> my_lock(dataset_mutex);
      error_state.push_back((error_state.back()));
    }else throw std::runtime_error("Can't modify state while set is exposed to multithreading!");
  }

  /**
   * @brief      Restores the previously stored state, if there is any
   */
  void pop_state(){
    if(!exposed_to_multithreading){
      std::lock_guard<std::mutex> my_lock(dataset_mutex);
      if(1 < error_state.size()) error_state.pop_back();
    }else throw std::runtime_error("Can't modify state while set is exposed to multithreading!");
  }

  /**
   * @brief      Returns the stored error for the sample under the index
   *
   * @param[in]  index  The index
   *
   * @return     The error.
   */
  sdouble32 get_error(uint32 index) const{
    if(error_state.back().sample_errors.size() > index)
      return error_state.back().sample_errors[index];
    else throw std::runtime_error("Sample index out of bounds!");
  }

  /**
   * @brief      Gets the overall stored error.
   *
   * @return     The sum of the errors for all of the samples.
   */
  sdouble32 get_error_sum() const{
    if(!exposed_to_multithreading){
      std::lock_guard<std::mutex> my_lock(dataset_mutex);
      return error_state.back().error_sum;
    } else throw std::runtime_error("Can't query error state while the set is exposd to multithreading");
  }

  /**
   * @brief      Gets the error average.
   *
   * @return     The error average.
   */
  sdouble32 get_error_avg() const{
    if(!exposed_to_multithreading){
      std::lock_guard<std::mutex> my_lock(dataset_mutex);
      return error_state.back().error_sum / dataset.get_number_of_label_samples();
    } else throw std::runtime_error("Can't query error state while the set is exposd to multithreading");
  }

  /**
   * @brief     Puts the set in a thread-safe state, enabling multi-threaded set access to the error_values vector, but
   *            disabling error_sum calculations(one of the main common part of the set).
   */
  void expose_to_multithreading(){
    exposed_to_multithreading = true;
  }

  /**
   * @brief     Restores the set to a non-thread-safe state, disabling multi-threaded set access to the error_values vector, but
   *            re-enabling error_sum calculations(one of the main common part of the set). Also re-calculates error value sum
   */
  void conceal_from_multithreading();

  /**
   * @brief      Provides access to the underlying @DataSet wrapper
   *
   * @return     A const reference to the underlying dataset wrapper
   */
  const RafkoDatasetWrapper& get_dataset(){
    return dataset;
  }

private:
  struct error_state_type{
    std::vector<sdouble32> sample_errors;
    sdouble32 error_sum;
  };
  static rafko_utilities::DataPool<sdouble32> common_datapool;

  rafko_mainframe::RafkoSettings& settings;
  const RafkoDatasetWrapper& dataset;
  std::vector<error_state_type> error_state;
  std::shared_ptr<rafko_net::CostFunction> cost_function;
  bool exposed_to_multithreading = false; /* basically decides whether or not error sum calculation is enabled. */
  mutable std::mutex dataset_mutex; /* when error sum calculation is enabled, the one common point of the dataset might be updated from different threads, so a std::mutex is required */
  const std::function<void(uint32)> error_calculation_lambda =  [this](uint32 thread_index){
    uint32 length = error_state.back().sample_errors.size() / settings.get_sqrt_of_solve_threads();
    uint32 start = length * thread_index;
    length = std::min(length, static_cast<uint32>(error_state.back().sample_errors.size() - start));
    accumulate_error_sum(start, length);
  };
  rafko_utilities::ThreadGroup error_calculation_threads;

  /**
   * @brief          Converting the @rafko_gym::DataSet message to vectors
   *
   * @param[in]      error_start    The starting index to read from in @error_state.sample_errors
   * @param[in]      errors_to_sum  The number of errors to add to @error_state.error_sum
   */
  void accumulate_error_sum(uint32 error_start, uint32 errors_to_sum);

};

} /* namespace rafko_gym */

#endif /* DATA_AGGREGATE_H */

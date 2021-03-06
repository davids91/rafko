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
#include <stdexcept>
#include <mutex>
#include <tuple>

#include "gen/common.pb.h"
#include "gen/sparse_net.pb.h"

#include "rafko_mainframe/models/service_context.h"

#include "rafko_gym/services/agent.h"
#include "rafko_utilities/services/thread_group.h"
#include "rafko_utilities/models/data_pool.h"
#include "sparse_net_library/models/cost_function.h"
#include "sparse_net_library/services/function_factory.h"

namespace rafko_gym{

using std::vector;
using std::shared_ptr;
using std::unique_ptr;
using std::move;
using std::mutex;
using std::tuple;

using rafko_mainframe::Service_context;
using rafko_utilities::ThreadGroup;
using sparse_net_library::SparseNet;
using sparse_net_library::Data_set;
using sparse_net_library::Function_factory;
using sparse_net_library::Cost_function;
using sparse_net_library::cost_functions;

/**
 * @brief      A Data set container complete with adaptive error statistics, which is
 *             not thread safe for the most part.
 *             It is possible to have more input samples, than label samples; In those cases
 *             the extra inputs are to be used to initialize the network before training.
 *             The data set consists of labels and inputs. Not every label has an input assigned to it,
 *             as there might be some additional inputs used to "prefill" a network, setting it up to be
 *             evaluated by the labels. It helps setting up an initial state for the training.
 *             The Dataset is built up of multiple sequences, each input and label in the sequence
 *             is of the same size and dimension. Each input and label can be of any size, albeit they must have
 *             the same size or every sample.
 *             ================================================
 *             Example of the structure:
 *             Dataset ( prefill 2, sequence size 6:
 *             - Sequence (sample) 1:
 *             - Inputs: [][][][][][]
 *             - Labels:     [][][][]
 *             - Sequence (sample) 2:
 *             - Inputs: [][][][][][]
 *             - Labels:     [][][][]
 *             - Sequence (sample) 3:
 *             - Inputs: [][][][][][]
 *             - Labels:     [][][][]
 *             - Sequence (sample) 4:
 *             - Inputs: [][][][][][]
 *             - Labels:     [][][][]
 *             ================================================
 *             Despite the above structure, for eligibility of paralellism, the inputs and labels are in a separate,
 *             contigous array.

 */
class Data_aggregate{
public:
  Data_aggregate(Service_context& service_context_, Data_set& samples_, shared_ptr<Cost_function> cost_function_)
  :  service_context(service_context_)
  ,  sequence_size(std::max(1u,samples_.sequence_size()))
  ,  input_samples(samples_.inputs_size() / samples_.input_size())
  ,  label_samples(samples_.labels_size() / samples_.feature_size())
  ,  prefill_sequences(static_cast<uint32>((samples_.inputs_size() - samples_.labels_size()) / (samples_.labels_size() / sequence_size)))
  ,  error_state(double_literal(1.0),{
       vector<sdouble32>(label_samples.size(),(double_literal(1.0)/label_samples.size())),
       double_literal(1.0)
     })
  ,  cost_function(cost_function_)
  ,  exposed_to_multithreading(false)
  ,  error_calculation_threads(service_context_.get_sqrt_of_solve_threads())
  {
    if(0 != (label_samples.size()%sequence_size))throw std::runtime_error("Sequence size doesn't match label number in Data set!");
    else fill(samples_);
  }

  Data_aggregate(
    Service_context& service_context_,
    vector<vector<sdouble32>>&& input_samples_, vector<vector<sdouble32>>&& label_samples_,
    shared_ptr<Cost_function> cost_function_, uint32 sequence_size_ = 1
  ): service_context(service_context_)
  ,  sequence_size(std::max(1u,sequence_size_))
  ,  input_samples(move(input_samples_))
  ,  label_samples(move(label_samples_))
  ,  prefill_sequences(static_cast<uint32>((input_samples.size() - label_samples.size()) / (label_samples.size() / sequence_size)))
  ,  error_state(double_literal(1.0),{
       vector<sdouble32>(label_samples.size(),(double_literal(1.0)/label_samples.size())),
       double_literal(1.0)
     })
  ,  cost_function(cost_function_)
  ,  exposed_to_multithreading(false)
  ,  error_calculation_threads(service_context_.get_sqrt_of_solve_threads())
  {
    if(0 != (label_samples.size()%sequence_size))throw std::runtime_error("Sequence size doesn't match label number in Data set!");
  }

  Data_aggregate(
    Service_context& service_context_,
    vector<vector<sdouble32>>&& input_samples_, vector<vector<sdouble32>>&& label_samples_,
    SparseNet& net, cost_functions the_function, uint32 sequence_size_ = 1
  ): service_context(service_context_)
  ,  sequence_size(std::max(1u,sequence_size_))
  ,  input_samples(move(input_samples_))
  ,  label_samples(move(label_samples_))
  ,  prefill_sequences(static_cast<uint32>((input_samples.size() - label_samples.size()) / (label_samples.size() / sequence_size)))
  ,  error_state(double_literal(1.0),{
       vector<sdouble32>(label_samples.size(),(double_literal(1.0)/label_samples.size())),
       double_literal(1.0)
     })
  ,  cost_function(Function_factory::build_cost_function(net, the_function, service_context_))
  ,  exposed_to_multithreading(false)
  ,  error_calculation_threads(service_context_.get_sqrt_of_solve_threads())
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
   * @brief      Same as @set_feature_for_label but in bulk
   *
   * @param[in]  neuron_data              The neuron data
   * @param[in]  neuron_buffer_index      The index of the outer neuron bufer to start evaluation from
   * @param[in]  raw_start_index          The raw start index inside the dataset labels; Meaning the index inside the labels array, which contains the samples(each with possible multiple labels in sequential order)
   * @param[in]  labels_to_evaluate       The labels to evaluate
   */
  void set_features_for_labels(
    const vector<vector<sdouble32>>& neuron_data,
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
    const vector<vector<sdouble32>>& neuron_data, uint32 neuron_buffer_index,
    uint32 sequence_start_index, uint32 sequences_to_evaluate,
    uint32 start_index_in_sequence, uint32 sequence_truncation
  ){
    vector<sdouble32>& resulting_errors = common_datapool.reserve_buffer(sequences_to_evaluate * get_sequence_size());
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
    const vector<vector<sdouble32>>& neuron_data, uint32 neuron_buffer_index,
    uint32 sequence_start_index, uint32 sequences_to_evaluate,
    uint32 start_index_in_sequence, uint32 sequence_truncation,
    vector<sdouble32>& tmp_data
  );

  /**
   * @brief      Sets the error values to the default value
   */
  void reset_errors(void){
    if(!exposed_to_multithreading){
      std::lock_guard<mutex> my_lock(dataset_mutex);
      for(sdouble32& sample_error : error_state.back().sample_errors)
        sample_error = (double_literal(1.0)/label_samples.size());
      error_state.back().error_sum = double_literal(1.0);
    }else throw std::runtime_error("Can't reset errors while set is exposed to multithreading!");
  }

  /**
   * @brief      Stores the current error values for later re-use
   */
  void push_state(void){
    if(!exposed_to_multithreading){
      std::lock_guard<mutex> my_lock(dataset_mutex);
      error_state.push_back((error_state.back()));
    }else throw std::runtime_error("Can't modify state while set is exposed to multithreading!");
  }

  /**
   * @brief      Restores the previously stored state, if there is any
   */
  void pop_state(void){
    if(!exposed_to_multithreading){
      std::lock_guard<mutex> my_lock(dataset_mutex);
      if(1 < error_state.size()) error_state.pop_back();
    }else throw std::runtime_error("Can't modify state while set is exposed to multithreading!");
  }

  /**
   * @brief      Gets an input sample from the set
   *
   * @param[in]  sample_index  The sample index
   *
   * @return     The input sample.
   */
  const vector<sdouble32>& get_input_sample(uint32 raw_input_index) const{
    if(input_samples.size() > raw_input_index)
      return input_samples[raw_input_index];
      else throw std::runtime_error("Input sample index out of bounds!");
  }

  /**
   * @brief      Gets a label sample from the set
   *
   * @param[in]  sample_index  The sample index
   *
   * @return     The label sample.
   */
  const vector<sdouble32>& get_label_sample(uint32 raw_label_index) const{
    if(label_samples.size() > raw_label_index)
      return label_samples[raw_label_index];
      else throw std::runtime_error("Label sample index out of bounds!");
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
  sdouble32 get_error_sum(void) const{
    if(!exposed_to_multithreading){
      std::lock_guard<mutex> my_lock(dataset_mutex);
      return error_state.back().error_sum;
    } else throw std::runtime_error("Can't query error state while the set is exposd to multithreading");
  }

  /**
   * @brief      Gets the erro average.
   *
   * @return     The erro average.
   */
  sdouble32 get_error_avg() const{
    if(!exposed_to_multithreading){
      std::lock_guard<mutex> my_lock(dataset_mutex);
      return error_state.back().error_sum / get_number_of_label_samples();
    } else throw std::runtime_error("Can't query error state while the set is exposd to multithreading");
  }

  /**
   * @brief      Gets the number of values present in the output
   *
   * @return     The feature size.
   */
  uint32 get_feature_size(void) const{
    return label_samples[0].size();
  }

  /**
   * @brief      Gets the number of raw input arrays stored in the pbject
   *
   * @return     The number of input samples.
   */
  uint32 get_number_of_input_samples(void) const{
    return input_samples.size();
  }

  /**
   * @brief      The number of raw label arrays stored in the object
   *
   * @return     The number of labels.
   */
  uint32 get_number_of_label_samples(void) const{
    return label_samples.size();
  }

  /**
   * @brief      Gets the number of sequences stored in the object. One sequence contains
   *             a number of input and label sample arrays. There might be more input arrays,
   *             than label arrays in one sequences. The difference is given by @get_prefill_inputs_number
   *
   * @return     The number of sequences.
   */
  uint32 get_number_of_sequences(void) const{
    return (get_number_of_label_samples() / sequence_size);
  }

  /**
   * @brief      Gets the size of one sequence
   *
   * @return     Number of consecutive datapoints that count as one sample.
   */
  uint32 get_sequence_size(void) const{
    return sequence_size;
  }

  /**
   * @brief      Gets the number of inputs to be used as initializing the network during a training run
   *
   * @return     The number of inputs to be used for network initialization during training
   */
  uint32 get_prefill_inputs_number(void) const{
    return prefill_sequences;
  }

  /**
   * @brief     Puts the set in a thread-safe state, enabling multi-threaded set access to the error_values vector, but
   *            disabling error_sum calculations(one of the main common part of the set).
   */
  void expose_to_multithreading(void){
    exposed_to_multithreading = true;
  }

  /**
   * @brief     Restores the set to a non-thread-safe state, disabling multi-threaded set access to the error_values vector, but
   *            re-enabling error_sum calculations(one of the main common part of the set). Also re-calculates error value sum
   */
  void conceal_from_multithreading(void);

private:
  struct error_state_type{
    vector<sdouble32> sample_errors;
    sdouble32 error_sum;
  };

  Service_context& service_context;
  uint32 sequence_size;
  vector<vector<sdouble32>> input_samples;
  vector<vector<sdouble32>> label_samples;
  uint32 prefill_sequences; /* Number of input sequences used only to create an initial state for the Neural network */
  vector<error_state_type> error_state;
  shared_ptr<Cost_function> cost_function;
  bool exposed_to_multithreading; /* basically decides whether or not error sum calculation is enabled. */
  mutable mutex dataset_mutex; /* when error sum calculation is enabled, the one common point of the dataset might be updated from different threads, so a mutex is required */
  const std::function<void(uint32)> error_calculation_lambda =  [this](uint32 thread_index){
    uint32 length = error_state.back().sample_errors.size() / service_context.get_sqrt_of_solve_threads();
    uint32 start = length * thread_index;
    length = std::min(length, static_cast<uint32>(error_state.back().sample_errors.size() - start));
    accumulate_error_sum(start, length);
  };
  ThreadGroup error_calculation_threads;

  static DataPool<sdouble32> common_datapool;

  /**
   * @brief      Converting the @Data_set message to vectors
   *
   * @param      samples  The data set to parse
   */
  void fill(Data_set& samples);

  /**
   * @brief          Converting the @Data_set message to vectors
   *
   * @param[in]      error_start    The starting index to read from in @error_state.sample_errors
   * @param[in]      errors_to_sum  The number of errors to add to @error_state.error_sum
   */
  void accumulate_error_sum(uint32 error_start, uint32 errors_to_sum);

};

} /* namespace rafko_gym */

#endif /* DATA_AGGREGATE_H */

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

#ifndef SPARSE_NET_APPROXIMIZER_H
#define SPARSE_NET_APPROXIMIZER_H

#include "sparse_net_global.h"

#include <cmath>

#include <mutex>

#include "gen/common.pb.h"

#include "sparse_net_library/models/data_aggregate.h"
#include "sparse_net_library/services/weight_updater.h"

#include "sparse_net_library/services/solution_solver.h"

namespace sparse_net_library{

using std::unique_ptr;
using std::make_unique;
using std::max;
using std::min;
using std::mutex;

/**
 * @brief      This class approximates gradients for a @Dataset and @Sparse_net.
 *             The approximated gradients are collected into one gradient fragment.
 */
class Sparse_net_approximizer{
public:
  Sparse_net_approximizer(
    SparseNet& neural_network, Data_aggregate& train_set_, Data_aggregate& test_set_,
    weight_updaters weight_updater_, Service_context& service_context
  );

  ~Sparse_net_approximizer(void){
    solvers.clear();
    if(nullptr == context.get_arena_ptr())
      delete net_solution;
  }
  Sparse_net_approximizer(const Sparse_net_approximizer& other) = delete;/* Copy constructor */
  Sparse_net_approximizer(Sparse_net_approximizer&& other) = delete; /* Move constructor */
  Sparse_net_approximizer& operator=(const Sparse_net_approximizer& other) = delete; /* Copy assignment */
  Sparse_net_approximizer& operator=(Sparse_net_approximizer&& other) = delete; /* Move assignment */

  /**
   * @brief      Checks if the data-set changed enough to be re-evaluated, and if it does, it evaluates it
   */
  void check(void){
    if(
      (loops_unchecked >= context.get_insignificant_iteration_count())
      ||(loops_unchecked > (train_set.get_error_sum()/context.get_step_size()))
      ||(loops_unchecked > (test_set.get_error_sum()/context.get_step_size()))
    ){
      /* calculate the error value for the current network in the testing and training datasets */
      evaluate();
      loops_unchecked = 0;
    }
  }

  /**
   * @brief      Evaluate the configured network on the given data set, which updates the stored error states of the set
   */
  void evaluate(void){
    evaluate(train_set,0,train_set.get_number_of_sequences(),0,train_set.get_sequence_size());
    evaluate(test_set,0,test_set.get_number_of_sequences(),0,test_set.get_sequence_size());
  }

  /**
   * @brief      Moves the network in a random direction, approximates the gradients based on that 
   *             and then reverts the the weight change
   */
  void collect_approximates_from_random_direction(void);

  /**
   * @brief      Moves the network in a direction based on induvidual weight gradients,
   *             approximates the gradients based on that and then reverts the the weight change
   */
  void collect_approximates_from_weight_gradients(void);

  /**
   * @brief      Move the network in the given direction, collect approximate gradient for it 
   *             and then reverts the weight change
   *
   * @param[in]  direction  The direction
   */
  void collect_approximates_from_direction(vector<sdouble32> direction);

  /**
   * @brief      Step the net in the opposite direction of the gradient slope
   */
  void collect_fragment(void);

  /**
   * @brief      Collects the approximate gradient of a single weight
   *
   * @param[in]  weight_index  The weight index to approximate for
   *
   * @return     The gradient approximation for the configured dataset
   */
  sdouble32 get_gradient_fragment(uint32 weight_index);

  /**
   * @brief      Applies the colleted gradient fragment to the configured network
   */
  void apply_fragment(void);

  /**
   * @brief      Discards the gradient fragment collected in the past
   */
  void discard_fragment(void){
    gradient_fragment = Gradient_fragment();
  }

  /**
   * @brief      Adds the given values to the stored fragment.
   *
   * @param[in]  weight_index             The weight index to give the value to
   * @param[in]  gradient_fragment_value  The value to give to the fragment
   */
  void add_to_fragment(uint32 weight_index, sdouble32 gradient_fragment_value);

  /**
   * @brief      Gets the previously collected gradient fragment.
   *
   * @return     The fragment.
   */
  const Gradient_fragment get_fragment(void){
    return gradient_fragment;
  }

  /**
   * @brief      Gives back the error of the configured Network based on the training dataset
   */
  sdouble32 get_train_error(void) const{
   return train_set.get_error_avg();
  }

  /**
   * @brief      Gives back the error of the configured Network based on the test set
   */
  sdouble32 get_test_error(void) const{
   return test_set.get_error_avg();
  }

  /**
   * @brief      Helper function to get the collected weight gradient fragment
   *
   * @return     Constant reference to the current weight gradients array
   */
  const Gradient_fragment& get_weight_gradient(void) const{
    return gradient_fragment;
  }

private:
  SparseNet& net;
  Service_context& context;
  Solution* net_solution;
  vector<unique_ptr<Solution_solver>> solvers;
  Data_aggregate& train_set;
  Data_aggregate& test_set;
  Gradient_fragment gradient_fragment;

  uint32 loops_unchecked;
  uint32 sequence_truncation;
  unique_ptr<Weight_updater> weight_updater;

  vector<thread> solve_threads; /* The threads to be started during optimizing the network */
  vector<vector<thread>> process_threads; /* The inner process thread to be started during net optimization */

  mutex dataset_mutex;

  /**
   * @brief      Evaluate the configured network on the given data set, which updates the stored error states of the set
   *
   * @param      data_set                 The data set
   * @param[in]  label_start_index        The raw label start
   * @param[in]  labels_to_eval           The labels to eval
   * @param[in]  start_index_in_sequence  The start index in sequence
   * @param[in]  sequence_truncation      The sequence truncation
   */
  void evaluate(
    Data_aggregate& data_set, uint32 label_start_index, uint32 labels_to_eval,
    uint32 start_index_in_sequence, uint32 sequence_truncation
  );

  /**
   * @brief      A thread to calcualte the error value for the given data set
   *
   * @param      data_set               The data set to calculate the error for
   * @param[in]  solve_thread_index     The index of the used thread
   * @param[in]  sequence_start_index   The starting sequence index to evaluate in this
   * @param[in]  sequences_to_evaluate  The sequences to evaluate
   * @param[in]  sequence_truncation    The sequence truncation
   */
  void evaluate_thread(
    Data_aggregate& data_set, uint32 solve_thread_index,
    uint32 sequence_start_index, uint32 sequences_to_evaluate,
    uint32 start_index_in_sequence, uint32 sequence_truncation
  );

  /**
   * @brief      Insert an element to the given position into the given field by
   *             first adding it to the end, and then reverse iterating and swapping elements
   *             until the desired position is reached
   *
   * @param      message_field  The message field
   * @param[in]  value          The value
   * @param[in]  position       The position
   */
  void insert_element_at_position(google::protobuf::RepeatedField<sdouble32>& message_field, sdouble32 value, uint32 position){
    *message_field.Add() = value;
    for(sint32 i(message_field.size() - 1); i > static_cast<sint32>(position); --i)
      message_field.SwapElements(i, i - 1);
  }

  /**
   * @brief      This function waits for the given threads to finish, ensures that every thread
   *             in the reference vector is finished, before it does.
   *
   * @param      calculate_threads  The calculate threads
   *//*!TODO: Find a better solution for these snippets */
  static void wait_for_threads(vector<thread>& calculate_threads){
    while(0 < calculate_threads.size()){
      if(calculate_threads.back().joinable()){
        calculate_threads.back().join();
        calculate_threads.pop_back();
      }
    }
  }
};

} /* namespace sparse_net_library */

#endif /* SPARSE_NET_APPROXIMIZER_H */
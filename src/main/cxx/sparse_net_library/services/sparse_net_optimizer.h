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

#ifndef SPARSE_NET_OPTIMIZER_H
#define SPARSE_NET_OPTIMIZER_H

#include "rafko_global.h"

#include <vector>
#include <memory>
#include <mutex>
#include <atomic>

#include "gen/common.pb.h"
#include "gen/solution.pb.h"
#include "sparse_net_library/models/cost_function.h"
#include "sparse_net_library/models/transfer_function.h"
#include "sparse_net_library/models/data_aggregate.h"
#include "sparse_net_library/services/weight_updater.h"
#include "sparse_net_library/services/solution_solver.h"

namespace sparse_net_library{

using std::vector;
using std::shared_ptr;
using std::unique_ptr;
using std::make_unique;
using std::min;
using std::max;
using std::mutex;
using std::atomic;

/**
 * @brief      An optimizer to train neural networks based on calculated gradients.
 */
class Sparse_net_optimizer{
public:
  Sparse_net_optimizer(
    SparseNet& neural_network, Data_aggregate& train_set_, Data_aggregate& test_set_,
    cost_functions the_function, weight_updaters weight_updater_, Service_context& service_context
  ): Sparse_net_optimizer(
    neural_network, train_set_, test_set_,
    Function_factory::build_cost_function(neural_network, the_function, service_context),
    weight_updater_, service_context
  ){ }

  Sparse_net_optimizer(
    SparseNet& neural_network, Data_aggregate& train_set_, Data_aggregate& test_set_,
    shared_ptr<Cost_function> the_function, weight_updaters weight_updater_, Service_context& service_context
  );

  ~Sparse_net_optimizer(void){
    solvers.clear();
    if(nullptr == context.get_arena_ptr())
      delete net_solution;
  }
  Sparse_net_optimizer(const Sparse_net_optimizer& other) = delete;/* Copy constructor */
  Sparse_net_optimizer(Sparse_net_optimizer&& other) = delete; /* Move constructor */
  Sparse_net_optimizer& operator=(const Sparse_net_optimizer& other) = delete; /* Copy assignment */
  Sparse_net_optimizer& operator=(Sparse_net_optimizer&& other) = delete; /* Move assignment */


  /**
   * @brief      Step the net in the opposite direction of the gradient slope
   */
  void step(void);

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
   * @brief      Helper function to get the weight gradient at the latest iteration
   *
   * @return     Constant reference to the current weight gradients array
   */
  vector<unique_ptr<atomic<sdouble32>>>& get_weight_gradient(void){
    return weight_gradient;
  }

private:
  SparseNet& net;
  Service_context& context;
  Transfer_function transfer_function;

  Solution* net_solution;
  vector<unique_ptr<Agent>> solvers;
  Data_aggregate& train_set;
  Data_aggregate& test_set;
  mutex set_mutex;

  uint32 loops_unchecked;
  uint32 sequence_truncation;
  Backpropagation_queue gradient_step; /* Defines the neuron order during back-propagation */
  shared_ptr<Cost_function> cost_function;
  unique_ptr<Weight_updater> weight_updater;

  vector<thread> solve_threads; /* The threads to be started during optimizing the network */
  vector<vector<thread>> process_threads; /* The inner process thread to be started during net optimization */
  vector<vector<vector<sdouble32>>> transfer_function_input; /* Copy of the Neurons data before the transfer function processed them in the structure of [Threads][Sequences(at most: @sequence_truncation)][Neurons] */
  vector<vector<unique_ptr<atomic<sdouble32>>>> error_values; /* Calculated error values: [Threads][Neurons] */
  vector<vector<vector<unique_ptr<atomic<sdouble32>>>>> weight_derivatives; /* Calculated derivatives for each weights [Threads][Sequences][Weights] */
  vector<unique_ptr<atomic<sdouble32>>> weight_gradient; /* calculated gradient values */
  vector<sdouble32> weight_gradient_vector; /* duplicate of @weight_gradient */

  /**
   * @brief      A thread to step the neural network forward
   *
   * @param[in]  solve_thread_index   The index of the solve thread the function is working in
   * @param[in]  samples_to_evaluate  The number of samples to evaluate
   */
  void step_thread(uint32 solve_thread_index, uint32 samples_to_evaluate);

  /**
   * @brief      Calculates the derivatives part of the gradient for each weight
   *             Starts @calculate_derivatives_thread-s simultaniously, almost equally dividing
   *             the number of output neurons to be calculated in one thread.
   *             The number of threads to be started depends on @Service_context::get_max_solve_threads
   *
   * @param[in]  solve_thread_index  The solve thread index
   * @param[in]  sequence_index      The sequence index
   * @param[in]  raw_inputs_index    The index of the input array the network was calculated with
   * @param[in]  raw_sample_index    The sample index in-between the sequences of the data-set
   */
  void calculate_derivatives(uint32 solve_thread_index, uint32 sequence_index, uint32 raw_inputs_index, uint32 raw_sample_index);

  /**
   * @brief      Calculates the output layer deviation from the given sample under @raw_sample_index.
   *             Starts @calculate_output_errors_thread-s simultaniously, almost equally dividing
   *             the number of output neurons to be calculated in one thread.
   *             The number of threads to be started depends on @Service_context::get_max_solve_threads
   *
   * @param[in]  solve_thread_index  The solve thread index
   * @param[in]  raw_inputs_index    The index of the input array the network was calculated with
   * @param[in]  raw_sample_index    The sample index in-between the sequences of the data-set
   */
  void calculate_output_errors(uint32 solve_thread_index, uint32 sequence_index, uint32 raw_inputs_index, uint32 raw_sample_index);

  /**
   * @brief      Propagates the error back to the hidden Neurons. Starts one thread for each Neuron
   *             up until @Service_context::get_max_solve_threads, waits for them to finish, then continues.
   *             The order in which the Neurons are processed is decided by @gradient_step.
   *
   * @param[in]  solve_thread_index  The solve thread index
   * @param[in]  sequence_index      The index of the currently evaluated sequence
   */
  void propagate_output_errors_back(uint32 solve_thread_index, uint32 sequence_index);

  /**
   * @brief      Calculates and accumulates weight gradients for the Neuron weights
   *             based on Neuron error values.
   *             Starts @calculate_output_errors_thread-s simultaniously, almost equally dividing
   *             the number of weights to be calculated in one thread.
   *             The number of threads to be started depends on @Service_context::get_max_solve_threads
   *
   * @param[in]  solve_thread_index  The solve thread index
   * @param[in]  sequence_index      The index of the sequence currently processed
   */
  void accumulate_weight_gradients(uint32 solve_thread_index, uint32 sequence_index);

  /**
   * @brief      Divides all weight gradients with the minibatch size, so the weights are being updated
   *             with an average value instead of an absolute one.
   *             Starts Service_Context::get_max_processing_threads, almost equally dividing the
   *             number of weights to process.
   */
  void normalize_weight_gradients(void);

  /**
   * @brief      The thread for calculating the derivative part for each weights
   *
   * @param[in]  solve_thread_index  The solve thread index
   * @param[in]  sequence_index      The index of the currently evaluated sequence
   * @param[in]  raw_inputs_index    The index of the input array the network was calculated with
   * @param[in]  raw_sample_index    The sample index in-between the sequences of the data-set
   * @param[in]  neuron_index        The neuron index to start calculatin output errors from
   * @param[in]  neuron_number       The number of neurons to include in this thread
   */
  void calculate_derivatives_thread(uint32 solve_thread_index, uint32 sequence_index, uint32 raw_inputs_index, uint32 raw_sample_index, uint32 neuron_index, uint32 neuron_number);

  /**
   * @brief      The thread call for calculating the output errors. MUltiple Output Neurons are calculated in one thread
   *             one after another.
   *             Starts @calculate_output_errors_thread-s simultaniously, almost equally dividing
   *             the number of weights to be calculated in one thread.
   *             The number of threads to be started depends on @Service_context::get_max_solve_threads
   *
   * @param[in]  solve_thread_index  The solve thread index
   * @param[in]  sequence_index      The index of the currently evaluated sequence
   * @param[in]  raw_inputs_index    The index of the input array the network was calculated with
   * @param[in]  raw_sample_index    The sample index in-between the sequences of the data-set
   * @param[in]  neuron_index        The neuron index to start calculatin output errors from
   * @param[in]  neuron_number       The number of neurons to include in this thread
   */
  void calculate_output_errors_thread(uint32 solve_thread_index, uint32 sequence_index, uint32 raw_inputs_index, uint32 raw_sample_index, uint32 neuron_index, uint32 neuron_number);

  /**
   * @brief      The thread used by @propagate_output_errors_back. For its every input node
   *             it propagates its error back, by copying its weigthed error, and multiplying it
   *             with that nodes derivative
   *
   * @param[in]  solve_thread_index  The solve thread index
   * @param[in]  sequence_index      The index of the currently evaluated sequence
   * @param[in]  neuron_index        The neuron index
   */
  void backpropagation_thread(uint32 solve_thread_index, uint32 sequence_index, uint32 neuron_index);

  /**
   * @brief      The thread used by @accumulate_weight_gradients, it iterates through
   *             the weights based on the given input arguments.
   *
   * @param[in]  solve_thread_index  The solve thread index
   * @param[in]  neuron_index        The starting neuron index
   * @param[in]  neuron_number       The number of neurpons to process in this thread
   */
  void accumulate_weight_gradients_thread(
    uint32 solve_thread_index, uint32 sequence_index, uint32 neuron_index, uint32 neuron_number
  );

  /**
   * @brief      The thread used by @normalize_weight_gradients, it iterates through
   *             the weights based on the given input arguments.
   *
   * @param[in]  weight_index   The weight index
   * @param[in]  weight_number  The weight number
   */
  void normalize_weight_gradients_thread(uint32 weight_index, uint32 weight_number);

  /**
   * @brief      Gets the derivative for the given arguments. Truncation considered: The gradients
   *             shall be calculated for as long as the truncation permits, and with any sequence above
   *             truncated sequences shall use the last calculated gradient.
   *
   * @param[in]  solve_thread_index  The index of the solve thread used
   * @param[in]  sequence_index      The sequence index
   * @param[in]  weight_index        The weight index
   * @param[in]  input_synapse       The input synapse
   *
   * @return     The derivative for.
   */
  sdouble32 get_derivative_for(uint32 solve_thread_index, uint32 sequence_index, uint32 weight_index, Input_synapse_interval input_synapse = Input_synapse_interval()){
    if(
      (context.get_max_solve_threads() <= solve_thread_index)
      ||(train_set.get_sequence_size() <= sequence_index)
      ||(net.weight_table_size() <= static_cast<sint32>(weight_index))
    )throw std::runtime_error("Weight index out of bounds!");
    return(*weight_derivatives[solve_thread_index][min(
      (sequence_truncation-1), (sequence_index - min(sequence_index, input_synapse.reach_past_loops()))
    )][weight_index]);
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

#endif /* SPARSE_NET_OPTIMIZER_H */

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

#include "sparse_net_global.h"
#include "gen/common.pb.h"
#include "gen/sparse_net.pb.h"
#include "gen/solution.pb.h"
#include "models/cost_function.h"
#include "models/transfer_function.h"
#include "models/data_aggregate.h"
#include "services/neuron_router.h"
#include "services/solution_builder.h"
#include "services/solution_solver.h"
#include "services/function_factory.h"
#include "services/backpropagation_queue_wrapper.h"
#include "services/updater_factory.h"

#include <vector>
#include <memory>

namespace sparse_net_library{

using std::vector;
using std::unique_ptr;
using google::protobuf::Arena;
using std::array;
using std::min;
using std::max;

class Sparse_net_optimizer{
public:
  Sparse_net_optimizer(
    SparseNet& neural_network, Data_aggregate& train_set_, Data_aggregate& test_set_,
    weight_updaters weight_updater_, Service_context service_context = Service_context()
  ): net(neural_network)
  ,  context(service_context)
  ,  transfer_function(context)
  ,  net_solution(Solution_builder().service_context(context).build(net))
  ,  solver(context.get_max_solve_threads(), Solution_solver(*net_solution, service_context))
  ,  train_set(train_set_)
  ,  test_set(test_set_)
  ,  loops_unchecked(50)
  ,  gradient_step(Backpropagation_queue_wrapper(neural_network)())
  ,  cost_function(Function_factory::build_cost_function(net, train_set.get_number_of_samples(), context))
  ,  solve_threads(0)
  ,  process_threads(context.get_max_solve_threads()) /* One queue for every solve thread */
  ,  neuron_data(context.get_max_solve_threads())
  ,  transfer_function_input(context.get_max_solve_threads())
  ,  transfer_function_output(context.get_max_solve_threads())
  ,  error_values(context.get_max_solve_threads())
  ,  weight_gradient(0)
  {
    (void)context.set_minibatch_size(max(1u,min(
      train_set.get_number_of_samples(),context.get_minibatch_size()
    )));
    solve_threads.reserve(context.get_max_solve_threads());
    for(uint32 threads = 0; threads < context.get_max_solve_threads(); ++threads){
      for(sint32 i = 0; i < net.neuron_array_size(); ++i)
        error_values[threads].push_back(std::make_unique<atomic<sdouble32>>());
      process_threads[threads].reserve(context.get_max_processing_threads());
      neuron_data[threads] = vector<sdouble32>(train_set.get_feature_size());
      transfer_function_input[threads] = vector<sdouble32>(train_set.get_feature_size());
      transfer_function_output[threads] = vector<sdouble32>(train_set.get_feature_size());
    }
    weight_gradient.reserve(net.weight_table_size());
    for(sint32 i = 0; i < net.weight_table_size(); ++i){
      weight_gradient.push_back(std::make_unique<atomic<sdouble32>>());
    }
    weight_updater = Updater_factory::build_weight_updater(net,weight_updater_,context);
  };

   /**
    * @brief      Step the net in the opposite direction of the gradient slope
    */
  void step(void);

  /**
   * @brief      Gives back the error of the configured Network based on the previous optimization step
   */
  sdouble32 get_train_error(){
   return train_set.get_error();
  }

  /**
   * @brief      Gives back the error of the configured Network based on the previous optimization step
   */
  sdouble32 get_test_error(){
   return test_set.get_error();
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
  Service_context context;
  Transfer_function transfer_function;

  unique_ptr<Solution> net_solution;
  vector<Solution_solver> solver;

  Data_aggregate& train_set;
  Data_aggregate& test_set;
  uint32 loops_unchecked;
  Backpropagation_queue gradient_step; /* Defines the neuron order during back-propagation */
  unique_ptr<Cost_function> cost_function;
  unique_ptr<Weight_updater> weight_updater;

  vector<thread> solve_threads; /* The threads to be started during optimizing the network */
  vector<vector<thread>> process_threads; /* The inner process thread to be started during net optimization */
  vector<vector<sdouble32>> neuron_data; /* Copy of the Neurons data for each solve thread */
  vector<vector<sdouble32>> transfer_function_input; /* Copy of the Neurons data for each solve thread */
  vector<vector<sdouble32>> transfer_function_output; /* Copy of the Neurons data for each solve thread */
  vector<vector<unique_ptr<atomic<sdouble32>>>> error_values; /* Calculated error values */
  vector<unique_ptr<atomic<sdouble32>>> weight_gradient; /* calculated gradient values */

  /**
   * @brief      A thread to step the neural network forward
   *
   * @param[in]  solve_thread_index   The index of the solve thread the function is working in
   * @param[in]  samples_to_evaluate  The number of samples to evaluate
   */
  void step_thread(uint32 solve_thread_index, uint32 samples_to_evaluate);

  /**
   * @brief      A Thread to evaluate a network on all given samples
   *
   * @param[in]  solve_thread_index   The index of the solve thread the function is working in
   * @param[in]  sample_start   The index of the sample to start evaluating from
   * @param[in]  samples_to_evaluate  Number of samples to be evaluated
   */
  void evaluate_thread(uint32 solve_thread_index, uint32 sample_start, uint32 samples_to_evaluate);

  /**
   * @brief      Calculates the output layer deviation from the given sample under @sample_index.
   *             Starts @calculate_output_errors_thread-s simultaniously, almost equally dividing 
   *             the number of output neurons to be calculated in one thread.
   *             The number of threads to be started depends on @Service_context::get_max_solve_threads
   *
   * @param[in]  solve_thread_index  The solve thread index
   * @param[in]  sample_index        The sample index
   */
  void calculate_output_errors(uint32 solve_thread_index, uint32 sample_index);

  /**
   * @brief      Propagates the error back to the hidden Neurons. Starts one thread for each Neuron
   *             up until @Service_context::get_max_solve_threads, waits for them to finish, then continues.
   *             The order in which the Neurons are processed is decided by @gradient_step.
   *
   * @param[in]  solve_thread_index  The solve thread index
   */
  void propagate_output_errors_back(uint32 solve_thread_index);

  /**
   * @brief      Calculates and accumulates weight gradients for the Neuron weights 
   *             based on Neuron error values.
   *             Starts @calculate_output_errors_thread-s simultaniously, almost equally dividing 
   *             the number of weights to be calculated in one thread.
   *             The number of threads to be started depends on @Service_context::get_max_solve_threads
   *
   * @param[in]  solve_thread_index  The solve thread index
   * @param[in]  sample_index        The sample index
   */
  void accumulate_weight_gradients(uint32 solve_thread_index, uint32 sample_index);

  /**
   * @brief      Divides all weight gradients with the minibatch size, so the weights are being updated 
   *             with an average value instead of an absolute one.
   *             Starts Service_Context::get_max_processing_threads, almost equally dividing the
   *             number of weights to process.
   */
  void normalize_weight_gradients(void);

  /**
   * @brief      The thread call for calculating the output errors. MUltiple Output Neurons are calculated in one thread
   *             one after another.
   *             Starts @calculate_output_errors_thread-s simultaniously, almost equally dividing 
   *             the number of weights to be calculated in one thread.
   *             The number of threads to be started depends on @Service_context::get_max_solve_threads
   *
   * @param[in]  solve_thread_index  The solve thread index
   * @param[in]  sample_index        The sample index
   * @param[in]  neuron_index        The neuron index to start calculatin output errors from
   * @param[in]  neuron_number       The number of neurons to include in this thread
   */
  void calculate_output_errors_thread(uint32 solve_thread_index, uint32 sample_index, uint32 neuron_index, uint32 neuron_number);

  /**
   * @brief      The thread used by @propagate_output_errors_back. For its every input node
   *             it propagates its error back, by copying its weigthed error, and multiplying it
   *             with that nodes derivative
   *
   * @param[in]  solve_thread_index  The solve thread index
   * @param[in]  neuron_index        The neuron index
   */
  void backpropagation_thread(uint32 solve_thread_index, uint32 neuron_index);

  /**
   * @brief      The thread used by @accumulate_weight_gradients, it iterates through 
   *             the weights based on the given input arguments.
   *
   * @param[in]  solve_thread_index  The solve thread index
   * @param[in]  sample_index        The sample index
   * @param[in]  neuron_index        The neuron index
   */
  void accumulate_weight_gradients_thread(uint32 solve_thread_index, uint32 sample_index, uint32 neuron_index);

  /**
   * @brief      The thread used by @normalize_weight_gradients, it iterates through 
   *             the weights based on the given input arguments.
   *
   * @param[in]  weight_index   The weight index
   * @param[in]  weight_number  The weight number
   */
  void normalize_weight_gradients_thread(uint32 weight_index, uint32 weight_number);

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

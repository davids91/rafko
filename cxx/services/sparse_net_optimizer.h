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

#ifndef SPARSE_NET_OPTIMIZER_H
#define SPARSE_NET_OPTIMIZER_H

#include "sparse_net_global.h"
#include "gen/common.pb.h"
#include "gen/sparse_net.pb.h"
#include "gen/solution.pb.h"
#include "models/cost_function.h"
#include "models/transfer_function.h"
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

class Sparse_net_optimizer{
public:
  Sparse_net_optimizer(
    SparseNet& neural_network, vector<vector<sdouble32>>& label_samples_,
    weight_updaters weight_updater_, Service_context service_context = Service_context()
  ): context(service_context)
  ,  label_samples(label_samples_)
  ,  net(neural_network)
  ,  transfer_function(context)
  ,  net_solution(*Solution_builder().service_context(context).build(net))
  ,  solver(context.get_max_solve_threads(), Solution_solver(net_solution, service_context))
  ,  gradient_step(Backpropagation_queue_wrapper(neural_network)())
  ,  cost_function(Function_factory::build_cost_function(net,label_samples))
  ,  neuron_router(net)
  ,  solve_threads(0)
  ,  process_threads(context.get_max_solve_threads()) /* One queue for every solve thread */
  ,  error_values(context.get_max_solve_threads())
  ,  weight_gradients(0)
  {
    solve_threads.reserve(context.get_max_solve_threads());
    for(uint32 threads = 0; threads < context.get_max_solve_threads(); ++threads){
      for(sint32 i = 0; i < net.neuron_array_size(); ++i)
        error_values[threads].push_back(std::make_unique<atomic<sdouble32>>());
      process_threads[threads].reserve(context.get_max_processing_threads());
    }
    for(sint32 i = 0; i < net.weight_table_size(); ++i)
      weight_gradients.push_back(std::make_unique<atomic<sdouble32>>());
    weight_updater = Updater_factory::build_weight_updater(net,weight_gradients,weight_updater_,context);
  };

   /**
    * @brief      Step the net in the opposite direction of the gradient slope
    *
    * @param      input_samples  The input samples to base the error function on
    * @param[in]  sample_size    The number of feature-label pairs considered to be 1 sample
    *                            this is important for recurrent Networks, ans the sample size
    *                            shall set how deep shall the Back propagation through time go,
    *                            and it also considers @smaple_size number of samples in the array
    *                            as 1 actual sample.
    */
  void step(vector<vector<sdouble32>>& input_samples, uint32 batch_size, uint32 sequence_size = 1);

  /**
   * @brief      Gives back the error of the configured Network based on the previous optimization step
   *
   * @return     Error value
   */
  sdouble32 get_last_error(){
   return last_error;
  }

private:
  Service_context context;
  vector<vector<sdouble32>>& label_samples;
  SparseNet& net;
  Transfer_function transfer_function;

  Solution net_solution;
  vector<Solution_solver> solver;

  Backpropagation_queue gradient_step;
  unique_ptr<Cost_function> cost_function;
  unique_ptr<Weight_updater> weight_updater;
  Neuron_router neuron_router;

  vector<thread> solve_threads;
  vector<vector<thread>> process_threads;
  vector<vector<unique_ptr<atomic<sdouble32>>>> error_values;
  vector<unique_ptr<atomic<sdouble32>>> weight_gradients;
  vector<sdouble32> gradient_values;
  atomic<sdouble32> last_error;

  void step_thread(vector<vector<sdouble32>>& input_samples, uint32 samples_to_evaluate, uint32 solve_thread_index);
  void calculate_output_errors(uint32 sample_index, uint32 solve_thread_index);
  void propagate_output_errors_back(uint32 solve_thread_index);
  void calculate_weight_gradients(vector<sdouble32>& input_sample, uint32 solve_thread_index);

  void calculate_output_errors_thread(uint32 sample_index, uint32 neuron_index, uint32 neuron_number, uint32 solve_thread_index);
  void backpropagation_thread(uint32 neuron_index, uint32 solve_thread_index);
  void calculate_weight_gradients_thread(vector<sdouble32>& input_sample, uint32 neuron_index, uint32 solve_thread_index);

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

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

#ifndef WEIGHT_UPDATER_H
#define WEIGHT_UPDATER_H

#include "sparse_net_global.h"

#include <thread>
#include <atomic>
#include <vector>

#include "gen/common.pb.h"
#include "gen/sparse_net.pb.h"
#include "gen/solution.pb.h"
#include "rafko_mainframe/models/service_context.h"

namespace sparse_net_library{

using std::ref;
using std::vector;
using std::atomic;
using std::thread;
using std::unique_ptr;

using rafko_mainframe::Service_context;

/**
 * @brief      Base implementation for updating weights for netowrks based on weight gradients
 */
class Weight_updater{
public:
  Weight_updater(
    SparseNet& sparse_net, Service_context& service_context, uint32 required_iterations_for_step_ = 1
  ): net(sparse_net)
  ,  context(service_context)
  ,  required_iterations_for_step(required_iterations_for_step_)
  ,  iteration(0)
  ,  finished(false)
  ,  current_velocity(sparse_net.weight_table_size(),double_literal(0.0))
  ,  calculate_threads()
  { calculate_threads.reserve(context.get_max_processing_threads()); };


  /**
   * @brief      The function to signal the weight updater that an iteration have started
   */
  void start(void){
    iteration = 0;
    finished = false;
  }

  /**
   * @brief      Do an iteration of weight updates. An actual weight update 
   *             shall count as valid when @required_iterations_for_step taken place.
   *
   * @param      gradients           The gradients
   * @param      solution            The solution
   */
  void iterate(vector<unique_ptr<atomic<sdouble32>>>& gradients, Solution& solution){
    calculate_velocity(gradients);
    update_weights_with_velocity();
    update_solution_with_weights(solution);
    iteration = (iteration + 1) % required_iterations_for_step;
    finished = (0 == iteration);
  }

  /**
   * @brief      Copies the referenced @SparseNet weights into the solution in the arguments
   *             It supposes that the solution is one already built, and it is built from
   *             the same @SparseNet referenced in the updater. Opens up a thread for every internal 
   *             neuron in the partial solution, up until a maximum of Service_context::get_max_processing_threads.
   *
   * @param      solution  The solution
   */
  void update_solution_with_weights(Solution& solution);

  /**
   * @brief      Tells if an iteration is at its valid state or not based on 
   *             he number of iterations since calling @start
   *
   * @return     True if finished, False otherwise.
   */
  bool is_finished() const{
    return finished;
  }

  sdouble32 get_current_velocity(uint32 weight_index) const{
    return current_velocity[weight_index];
  }

  const vector<sdouble32>& get_current_velocity() const{
    return current_velocity;
  }

  virtual ~Weight_updater() = default;

protected:
  SparseNet& net;
  Service_context& context;
  const uint32 required_iterations_for_step;
  uint32 iteration;
  bool finished;
  vector<sdouble32> current_velocity;
  
  /**
   * @brief      Gets the new value for one weight based on the gradients.
   *             More complex weight updaters should overwirte this function,
   *             as it is the basis of updating weights.
   *
   * @param[in]  weight_index        The weight index
   *
   * @return     The new weight.
   */
  sdouble32 get_new_weight(uint32 weight_index){
    return(net.weight_table(weight_index) - get_current_velocity(weight_index));
  }

  /**
   * @brief      Returns with a velocity value for a weight based on the provided gradients
   *
   * @param[in]  weight_index  The weight index
   * @param[in]  gradients     The gradients
   *
   * @return     The new velocity.
   */
  sdouble32 get_new_velocity(uint32 weight_index, const vector<unique_ptr<atomic<sdouble32>>>& gradients){
    return (*gradients[weight_index] * context.get_step_size());
  }

private:
  vector<thread> calculate_threads;

  /**
   * @brief      Calculates and stroes the required velocity for a weight based on the provided gradients
   *
   * @param[in]  gradients  The gradients array of size equal to the weights of the configured net
   */
  void calculate_velocity(const vector<unique_ptr<atomic<sdouble32>>>& gradients);

  /**
   * @brief      The function to update every weight of the referenced @SparseNet
   *             based on the values provided by @get_new_weight.
   *             It starts multiple threads, dividing almost equally the number of weights
   *             to be updated in each thread.
   */
  void update_weights_with_velocity(void);

  /**
   * @brief      A thread to update the weights of the @SpraseNet, called by @update_weights_with_velocity
   *
   * @param[in]  weight_index        The weight index
   * @param[in]  weight_number       The weight number
   */
  void calculate_velocity_thread(const vector<unique_ptr<atomic<sdouble32>>>& gradients, uint32 weight_index, uint32 weight_number){
    for(uint32 weight_iterator = 0; weight_iterator < weight_number; ++weight_iterator){
      current_velocity[weight_index + weight_iterator] = get_new_velocity(weight_index + weight_iterator, gradients);
    }
  }

  /**
   * @brief      A thread to calculate the latest velocity based on the gradients
   *
   * @param      gradients      The gradients
   * @param[in]  weight_index   The weight index
   * @param[in]  weight_number  The weight number
   */
  void update_weight_with_velocity(uint32 weight_index, uint32 weight_number){
    for(uint32 weight_iterator = 0; weight_iterator < weight_number; ++weight_iterator){
      net.set_weight_table(
        weight_index + weight_iterator, get_new_weight(weight_index + weight_iterator)
      );
    }
  }

  /**
   * @brief      A thread to copy a weight synapse from the referenced @SparseNet
   *             into a partial solution.
   *
   * @param[in]  inner_neuron_index                The inner neuron index
   * @param      partial                           The partial
   * @param[in]  neuron_weight_synapse_starts      The neuron weight synapse starts
   * @param[in]  inner_neuron_weight_index_starts  The inner neuron weight index starts
   */
  void copy_weight_to_solution(
    uint32 neuron_index, uint32 inner_neuron_index, Partial_solution& partial, uint32 inner_neuron_weight_index_starts
  );

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

#endif /* WEIGHT_UPDATER_H */
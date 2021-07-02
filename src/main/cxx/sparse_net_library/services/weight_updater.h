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

#include "rafko_global.h"

#include <vector>

#include "gen/solution.pb.h"
#include "rafko_utilities/services/thread_group.h"
#include "rafko_mainframe/models/service_context.h"

namespace sparse_net_library{

using std::ref;
using std::vector;

using rafko_utilities::ThreadGroup;
using rafko_mainframe::Service_context;

/**
 * @brief      Base implementation for updating weights for netowrks based on weight gradients
 */
class Weight_updater{
public:
  Weight_updater(
    SparseNet& sparse_net, Service_context& service_context_, uint32 required_iterations_for_step_ = 1
  ): net(sparse_net)
  ,  service_context(service_context_)
  ,  required_iterations_for_step(required_iterations_for_step_)
  ,  weights_to_do_in_one_thread(1u + static_cast<uint32>(net.weight_table_size()/service_context.get_max_solve_threads()))
  ,  iteration(0)
  ,  finished(false)
  ,  current_velocity(sparse_net.weight_table_size(),double_literal(0.0))
  ,  execution_threads(service_context.get_max_solve_threads())
  { };


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
  void iterate(const vector<sdouble32>& gradients, Solution& solution){
    calculate_velocity(gradients);
    update_weights_with_velocity();
    update_solution_with_weights(solution);
    iteration = (iteration + 1) % required_iterations_for_step;
    finished = (0 == iteration);
  }

  /**
   * @brief      Copies the weights in the stored @SparseNet reference into the provided solution.
   *             It supposes that the solution is one already built, and it is built from
   *             the same @SparseNet referenced in the updater. Uses a different thread for every partial solution.
   *
   * @param      solution  The solution
   */
  void update_solution_with_weights(Solution& solution) const;

  /**
   * @brief      Copies the weights in the stored @SparseNet reference into the provided solution.
   *             It supposes that the solution is one already built, and it is built from
   *             the same @SparseNet referenced in the updater. Uses a different thread for every partial solution.
   *
   * @param      solution       The solution
   * @param[in]  weight_index   The index of the weight to take over fro the @SparseNet
   */
  void update_solution_with_weight(Solution& solution, uint32 weight_index) const;

  /**
   * @brief      Tells if an iteration is at its valid state or not based on
   *             he number of iterations since calling @start
   *
   * @return     True if finished, False otherwise.
   */
  bool is_finished() const{
    return finished;
  }

  /**
   * @brief      returns the current stored velocity under the given weight index
   *
   * @param[in]  weight_index  The weight index to query
   *
   * @return     The current velocity.
   */
  sdouble32 get_current_velocity(uint32 weight_index) const{
    return current_velocity[weight_index];
  }

  /**
   * @brief      Gets the stored velocity vector which is the basis for updating the weights.
   *
   * @return     The current velocity.
   */
  const vector<sdouble32>& get_current_velocity() const{
    return current_velocity;
  }

  virtual ~Weight_updater() = default;

protected:
  SparseNet& net;
  Service_context& service_context;
  const uint32 required_iterations_for_step;
  const uint32 weights_to_do_in_one_thread;
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
  sdouble32 get_new_weight(uint32 weight_index) const{
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
  sdouble32 get_new_velocity(uint32 weight_index, const vector<sdouble32>& gradients) const{
    return (gradients[weight_index] * service_context.get_step_size());
  }

private:
  ThreadGroup execution_threads;

  /**
   * @brief      Calculates and stroes the required velocity for a weight based on the provided gradients
   *
   * @param[in]  gradients  The gradients array of size equal to the weights of the configured net
   */
  void calculate_velocity(const vector<sdouble32>& gradients);

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
  void calculate_velocity_thread(const vector<sdouble32>& gradients, uint32 weight_index, uint32 weight_number);

  /**
   * @brief      A thread to calculate the latest velocity based on the gradients
   *
   * @param      gradients      The gradients
   * @param[in]  weight_index   The weight index
   * @param[in]  weight_number  The weight number
   */
  void update_weight_with_velocity(uint32 weight_index, uint32 weight_number);

  /**
   * @brief      Copies the weights of a Neuron from the referenced @SparseNet
   *             into the partial solution reference provided as an argument.
   *             The @Partial_solution must be built from the SparseNet, as a pre-requisite.
   *
   * @param[in]  neuron_index                      The index of the Neuron inside the @SparsNet
   * @param[in]  inner_neuron_index                The index of the Neuron inside the @Partial_solution
   * @param      partial                           The partial solution to update
   * @param[in]  inner_neuron_weight_index_starts  The index in the weight table (of the @Partial_solution) where the inner neuron weights start
   */
  void copy_weights_of_neuron_to_partial_solution(
    uint32 neuron_index, uint32 inner_neuron_index,
    Partial_solution& partial, uint32 inner_neuron_weight_index_starts
  ) const;

  /**
   * @brief      Copies the weight of a Neuron under the given index from the referenced @SparseNets
   *             weight table into the partial solution reference provided as an argument.
   *             The @Partial_solution must be built from the SparseNet, as a pre-requisite.
   *
   * @param[in]  neuron_index                      The index of the Neuron inside the @SparsNet
   * @param[in]  weight_index                      The index of the Neurons weight inside the weight table of @SparsNet
   * @param[in]  inner_neuron_index                The index of the Neuron inside the @Partial_solution
   * @param      partial                           The partial solution to update
   * @param[in]  inner_neuron_weight_index_starts  The index in the weight table (of the @Partial_solution) where the inner neuron weights start
   */
  void copy_weight_of_neuron_to_partial_solution(
    uint32 neuron_index, uint32 weight_index, uint32 inner_neuron_index,
    Partial_solution& partial, uint32 inner_neuron_weight_index_starts
  ) const;
};

} /* namespace sparse_net_library */

#endif /* WEIGHT_UPDATER_H */

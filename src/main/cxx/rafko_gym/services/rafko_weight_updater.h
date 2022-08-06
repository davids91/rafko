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

#ifndef RAFKO_WEIGHT_UPDATER_H
#define RAFKO_WEIGHT_UPDATER_H

#include "rafko_global.h"

#include <vector>
#include <utility>
#include <unordered_map>
#include <mutex>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/solution.pb.h"
#include "rafko_utilities/services/thread_group.h"
#include "rafko_mainframe/models/rafko_settings.h"

namespace rafko_gym {

/**
 * @brief      Base implementation for updating weights for netowrks based on weight gradients
 */
class RAFKO_FULL_EXPORT RafkoWeightUpdater{
public:
  RafkoWeightUpdater(rafko_net::RafkoNet& rafko_net, const rafko_mainframe::RafkoSettings& settings_, std::uint32_t required_iterations_for_step_ = 1u)
  : network(rafko_net)
  , settings(settings_)
  , required_iterations_for_step(required_iterations_for_step_)
  , weights_to_do_in_one_thread(1u + static_cast<std::uint32_t>(network.weight_table_size()/settings.get_max_solve_threads()))
  , current_velocity(network.weight_table_size(),(0.0))
  , execution_threads(settings.get_max_solve_threads())
  {
  }


  /**
   * @brief      The function to signal the weight updater that an iteration have started
   */
  constexpr void start(){
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
  virtual void iterate(const std::vector<double>& gradients);

  /**
   * @brief      Tells if an iteration is at its valid state or not based on
   *             he number of iterations since calling @start
   *
   * @return     True if finished, False otherwise.
   */
  constexpr bool is_finished() const{
    return finished;
  }

  /**
   * @brief      returns the current stored velocity under the given weight index
   *
   * @param[in]  weight_index  The weight index to query
   *
   * @return     The current velocity.
   */
  virtual double get_current_velocity(std::uint32_t weight_index) const{
    return current_velocity[weight_index];
  }

  /**
   * @brief      Gets the stored velocity vector which is the basis for updating the weights.
   *
   * @return     The current velocity.
   */
  virtual const std::vector<double>& get_current_velocity() const{
    return current_velocity;
  }

  virtual ~RafkoWeightUpdater() = default;

protected:
  rafko_net::RafkoNet& network;
  const rafko_mainframe::RafkoSettings& settings;
  const std::uint32_t required_iterations_for_step;
  const std::uint32_t weights_to_do_in_one_thread;
  std::uint32_t iteration = 0u;
  bool finished = false;
  std::vector<double> current_velocity;

  /**
   * @brief      Gets the new value for one weight based on the velocity.
   *
   * @param[in]  weight_index        The weight index
   *
   * @return     The new weight.
   */
  virtual double get_new_weight(std::uint32_t weight_index) const{
    return(network.weight_table(weight_index) + get_current_velocity(weight_index));
  }

  /**
   * @brief      Returns with a velocity value for a weight based on the provided gradients
   *
   * @param[in]  weight_index  The weight index
   * @param[in]  gradients     The gradients
   *
   * @return     The new velocity.
   */
  virtual double get_new_velocity(std::uint32_t weight_index, const std::vector<double>& gradients) const{
    return (-gradients[weight_index] * settings.get_learning_rate());
  }

private:
  rafko_utilities::ThreadGroup execution_threads;
  mutable std::mutex reference_mutex;

  /**
   * @brief      Calculates and stroes the required velocity for a weight based on the provided gradients
   *
   * @param[in]  gradients  The gradients array of size equal to the weights of the configured net
   */
  void calculate_velocity(const std::vector<double>& gradients);

  /**
   * @brief      The function to update every weight of the referenced @RafkoNet
   *             based on the values provided by @get_new_weight.
   *             It starts multiple threads, dividing almost equally the number of weights
   *             to be updated in each thread.
   */
  void update_weights_with_velocity();

  /**
   * @brief      A thread to calculate the latest velocity based on the gradients
   *
   * @param      gradients      The gradients
   * @param[in]  weight_index   The weight index
   * @param[in]  weight_number  The weight number
   */
  void update_weight_with_velocity(std::uint32_t weight_index, std::uint32_t weight_number);

  /**
   * @brief      Copies the weights of a Neuron from the referenced @RafkoNet
   *             into the partial solution reference provided as an argument.
   *             The @PartialSolution must be built from the RafkoNet, as a pre-requisite.
   *
   * @param[in]  neuron_index                      The index of the Neuron inside the @SparsNet
   * @param      partial                           The partial solution to update
   * @param[in]  inner_neuron_weight_index_starts  The index in the weight table (of the @PartialSolution) where the inner neuron weights start
   */
  void copy_weights_of_neuron_to_partial_solution(
    std::uint32_t neuron_index, rafko_net::PartialSolution& partial, std::uint32_t inner_neuron_weight_index_starts
  ) const;
};

} /* namespace rafko_gym */

#endif /* RAFKO_WEIGHT_UPDATER_H */

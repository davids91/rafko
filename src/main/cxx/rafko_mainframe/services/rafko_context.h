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

#ifndef RAFKO_CONTEXT_H
#define RAFKO_CONTEXT_H

#include "rafko_global.h"

#include <memory>
#include <vector>
#include <functional>

#include "rafko_utilities/services/thread_group.h"

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/training.pb.h"
#include "rafko_gym/models/rafko_dataset_cost.h"
#include "rafko_gym/models/rafko_dataset_wrapper.h"
#include "rafko_gym/models/rafko_environment.h"
#include "rafko_gym/models/rafko_agent.h"
#include "rafko_gym/services/rafko_weight_updater.h"

namespace rafko_mainframe{

/**
 * @brief      The interface for the main context of the Rafko Deep learning service. It encapsulates a Network as its central
 *             point, and provides methods to refine it and solve it.
 */
class RAFKO_FULL_EXPORT RafkoContext{
public:
  virtual ~RafkoContext() = default;

  /**
   * @brief          Accepts an environment to base network evaluation on top of and takes ownership of it!
   *
   * @param[in]      environment    An environment ready to be moved inside the context
   */
  virtual void set_environment(std::shared_ptr<rafko_gym::RafkoEnvironment> environment_) = 0;

  /**
   * @brief          Accepts an objective function to base network evaluation on top of and takes ownership of it!
   *
   * @param[in]      objective    An objective function ready to be moved inside the context
   */
  virtual void set_objective(std::shared_ptr<rafko_gym::RafkoObjective> objective_) = 0;

  /**
   * @brief          Modifies a weight of the stored Netowrk
   *
   * @param[in]      weight_index     The index inside the networks weight table to be modified
   * @param[in]      weight_value     The value to set the new weight to
   */
  virtual void set_network_weight(uint32 weight_index, sdouble32 weight_value) = 0;

  /**
   * @brief          Sets every weight of the stored Network directly
   *
   * @param[in]      weights     The values of the weights to be set
   */
  virtual void set_network_weights(const std::vector<sdouble32>& weights) = 0;

  /**
   * @brief          Applies a weight change based on the provided weights vector
   *
   * @param[in]      weight_delta     The values to update the weights with
   */
  virtual void apply_weight_update(const std::vector<sdouble32>& weight_delta) = 0;

  /**
   * @brief      Evaluates installed agents and returns with its error/fittness value
   *
   * @return     The resulting error/fitness value summary of the evaluation
   */
  virtual sdouble32 full_evaluation() = 0;

  /**
   * @brief          Evaluates installed agents in a stochastic manner and returns with its error/fittness value
   *
   * @param[in]      to_seed        A helper value to make Stochastic evaluation deterministicly reproducible
   * @param[in]      seed_value     A helper value to make Stochastic evaluation deterministicly reproducible
   * @return         The resulting error/fitness value summary of the evaluation
   */
  virtual sdouble32 stochastic_evaluation(bool to_seed = false, uint32 seed_value = 0u) = 0;

  /**
   * @brief      For the provided input, return the result of the neural network
   *
   * @param[in]      input                  The input data to be taken
   * @param[in]      reset_neuron_data      should the internal memory of the solver is to be resetted before solving the neural network
   * @param[in]      thread_index           The index of thread the solution is to be running from
   *
   * @return         The output values of the network result
   */
  virtual rafko_utilities::ConstVectorSubrange<> solve(
    const std::vector<sdouble32>& input,
    bool reset_neuron_data = true, uint32 thread_index = 0
  ) = 0;

  /**
   * @brief      Saves the context state
   */
  virtual void push_state() = 0;

  /**
   * @brief      Restores the previously stored context state
   */
  virtual void pop_state() = 0;

  /**
   * @brief       Provides access to the settings
   *
   * @return      a reference of the contained settings
   */
  virtual rafko_mainframe::RafkoSettings& expose_settings() = 0;

  /**
   * @brief       Provides the reference the context builds over
   *
   * @return      a reference of the referenced network
   */
  virtual rafko_net::RafkoNet& expose_network() = 0;
};

} /* namespace rafko_mainframe */

#endif /* RAFKO_CONTEXT_H */

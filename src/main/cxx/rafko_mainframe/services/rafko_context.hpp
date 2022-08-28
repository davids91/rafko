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

#include "rafko_global.hpp"

#include <memory>
#include <vector>
#include <functional>

#include "rafko_utilities/services/thread_group.hpp"

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/training.pb.h"
#include "rafko_gym/models/rafko_cost.hpp"
#include "rafko_gym/models/rafko_dataset_wrapper.hpp"
#include "rafko_gym/models/rafko_environment.hpp"
#include "rafko_gym/models/rafko_agent.hpp"
#include "rafko_gym/services/rafko_weight_updater.hpp"
#include "rafko_mainframe/models/rafko_settings.hpp"

namespace rafko_mainframe{

/**
 * @brief      The interface for the main context of the Rafko Deep learning service. It encapsulates a Network as its central
 *             point, and provides methods to refine it and solve it.
 */
class RAFKO_FULL_EXPORT RafkoContext{

  /**
   * @brief     Constructs an arena in case the provided settings doesn't contain any
   *
   * @param     settings    The @RafkoSettings instance to check for an existing arena implementation
   *
   * @return    The pointer to the arena should the @RafkoSettings instance not contain it.
   */
  static std::unique_ptr<google::protobuf::Arena> initialize_arena(rafko_mainframe::RafkoSettings& settings){
    if(nullptr == settings.get_arena_ptr())
      return std::make_unique<google::protobuf::Arena>();
      else return {};
  }

  static std::shared_ptr<rafko_mainframe::RafkoSettings> initialize_settings(std::shared_ptr<rafko_mainframe::RafkoSettings> candidate){
    if(candidate)
      return candidate;
      else return std::make_shared<rafko_mainframe::RafkoSettings>();
  }

public:
  RafkoContext(std::shared_ptr<rafko_mainframe::RafkoSettings> settings = {})
  : m_settings(initialize_settings(settings))
  , m_arena(initialize_arena(*settings))
  {
    if(m_arena)m_settings->set_arena_ptr(m_arena.get());
  }

  virtual ~RafkoContext() = default;

  /**
   * @brief          Accepts an environment to base network evaluation on top of and takes ownership of it!
   *
   * @param[in]      environment    An environment ready to be moved inside the context
   */
  virtual void set_environment(std::shared_ptr<rafko_gym::RafkoEnvironment> environment) = 0;

  /**
   * @brief          Accepts an objective function to base network evaluation on top of and takes ownership of it!
   *
   * @param[in]      objective    An objective function ready to be moved inside the context
   */
  virtual void set_objective(std::shared_ptr<rafko_gym::RafkoObjective> objective) = 0;

  /**
   * @brief          Accepts a weight updater type to make handle the weight updates
   *
   * @param[in]      objective    An objective function ready to be moved inside the context
   */
  virtual void set_weight_updater(rafko_gym::Weight_updaters updater) = 0;

  /**
   * @brief          Updates the stored solution based on the network reference
   */
  virtual void refresh_solution_weights() = 0;

  /**
   * @brief          Modifies a weight of the stored Network
   *
   * @param[in]      weight_index     The index inside the networks weight table to be modified
   * @param[in]      weight_value     The value to set the new weight to
   */
  virtual void set_network_weight(std::uint32_t weight_index, double weight_value) = 0;

  /**
   * @brief          Sets every weight of the stored Network directly
   *
   * @param[in]      weights     The values of the weights to be set
   */
  virtual void set_network_weights(const std::vector<double>& weights) = 0;

  /**
   * @brief          Applies a weight change based on the provided weights vector
   *
   * @param[in]      weight_delta     The values to update the weights with
   */
  virtual void apply_weight_update(const std::vector<double>& weight_delta) = 0;

  /**
   * @brief      Evaluates installed agents and returns with its error/fittness value
   *
   * @return     The resulting error/fitness value summary of the evaluation
   */
  virtual double full_evaluation() = 0;

  /**
   * @brief          Evaluates installed agents in a stochastic manner and returns with its error/fittness value
   *
   * @param[in]      to_seed        A helper value to make Stochastic evaluation deterministicly reproducible
   * @param[in]      seed_value     A helper value to make Stochastic evaluation deterministicly reproducible
   * @return         The resulting error/fitness value summary of the evaluation
   */
  virtual double stochastic_evaluation(bool to_seed = false, std::uint32_t seed_value = 0u) = 0;

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
    const std::vector<double>& input,
    bool reset_neuron_data = true, std::uint32_t thread_index = 0
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
protected:
  std::shared_ptr<rafko_mainframe::RafkoSettings> m_settings;
  std::unique_ptr<google::protobuf::Arena> m_arena;
};

} /* namespace rafko_mainframe */

#endif /* RAFKO_CONTEXT_H */

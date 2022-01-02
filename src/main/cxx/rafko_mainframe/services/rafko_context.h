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
   * @brief          Accepts an environment to base network evaluation on top of.
   *
   * @param[in]      environment    An environment ready to be moved inside the context
   */
  virtual void set_environment(rafko_gym::RafkoEnvironment&& environment) = 0;

  virtual const rafko_gym::RafkoEnvironment& get_environment() = 0;

  /**
   * @brief          Accepts an objective function to base network evaluation on top of.
   *
   * @param[in]      objective    An objective function ready to be moved inside the context
   */
  virtual void set_objective_function(rafko_gym::RafkoObjective&& objective) = 0;

  virtual const rafko_gym::RafkoObjective& get_objective() = 0;

  /**
   * @brief          Accepts a weight updater to base network evaluation on top of.
   *
   * @param[in]      weight_updater    A weight updater ready to be moved inside the context
   */
  virtual void set_weight_updater(rafko_gym::RafkoWeightUpdater&& weight_updater) = 0;

  virtual rafko_gym::RafkoWeightUpdater& expose_weight_updater() = 0;

  /**
   * @brief      Evaluates installed agents and returns with its error/fittness value
   *
   * @return     The resulting error/fitness value summary of the evaluation
   */
  virtual sdouble32 full_evaluation() = 0;

  /**
   * @brief          Evaluates installed agents in a stochastic manner and returns with its error/fittness value
   *
   * @param[in]      seed    A helper value to make Stochastic evaluation deterministicly reproducible
   * @return         The resulting error/fitness value summary of the evaluation
   */
  virtual sdouble32 stochastic_evaluation(uint32 seed = 0u) = 0;

  /**
   * @brief      Saves the context state
   */
  virtual void push_state() = 0;

  /**
   * @brief      Restores the previously stored context state
   */
  virtual void pop_state() = 0;

  virtual rafko_mainframe::RafkoSettings& expose_settings() = 0;
  virtual sdouble32 get_current_fitness() = 0;

  virtual rafko_net::RafkoNet& expose_network() = 0;
};

} /* namespace rafko_mainframe */

#endif /* RAFKO_CONTEXT_H */

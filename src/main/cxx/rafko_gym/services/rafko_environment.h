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

#ifndef RAFKO_ENVIRONMENT_H
#define RAFKO_ENVIRONMENT_H

#include "rafko_global.h"

#include "rafko_gym/services/rafko_agent.h"

namespace RAFKO_FULL_EXPORT rafko_gym{

/**
 * @brief      A class representing an environment, producing fitness/error value. Error values are negative, while fittness
 *             values are positive
 */
class RafkoEnvironment{
public:
  /**
   * @brief      Evaluates the given agent and returns with its error/fittness value
   *
   * @param[in]      agent    The actor to be evaluated in the current environment
   * @return         The resulting error/fitness value summary of the evaluation
   */
  virtual sdouble32 full_evaluation(RafkoAgent& agent) = 0;

  /**
   * @brief      Evaluates the given agent in a stochastic manner and returns with its error/fittness value
   *
   * @param[in]      agent    The actor to be evaluated in the current environment
   * @param[in]      seed    A helper value to make Stochastic evaluation deterministicly reproducible
   * @return         The resulting error/fitness value summary of the evaluation
   */
  virtual sdouble32 stochastic_evaluation(RafkoAgent& agent, uint32 seed = 0u) = 0;

  /**
   * @brief      Saves the RafkoEnvironment state
   */
  virtual void push_state(void) = 0;

  /**
   * @brief      Provides the last measured training fitness value
   *
   * @return     The resulting error/fitness value summary of the evaluation
   */
  virtual sdouble32 get_training_fitness(void) = 0;

  /**
   * @brief      Provides the last measured testing fitness value
   *
   * @return     The resulting error/fitness value summary of the evaluation
   */
  virtual sdouble32 get_testing_fitness(void) = 0;

  /**
   * @brief      Restores the previously stored environment state
   */
  virtual void pop_state(void) = 0;

  virtual ~RafkoEnvironment(void) = default;
};

} /* namespace rafko_gym */

#endif /* RAFKO_ENVIRONMENT_H */

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

#ifndef AGENT_H
#define AGENT_H

#include "rafko_global.h"

#include "gen/solution.pb.h"
#include "sparse_net_library/models/data_ringbuffer.h"

namespace sparse_net_library{

/**
 * @brief      This class serves as a base for reinforcement learning agent, which provides output data
 *              based on different inputs
 */
class Agent{
public:

  /**
   * @brief      Solves the Solution given in the constructor, considering the previous runs
   *
   * @param[in]      input    The input data to be taken
   * @param          output   The used Output data to write the results to
   */
  virtual void solve(const vector<sdouble32>& input, DataRingbuffer& output) const = 0;

  /**
   * @brief      Provide the underlying solution the solver is built to solve.
   *
   * @return     A const reference to the solution the solver is aiming to solve
   */
  virtual const Solution& get_solution(void) const = 0;

  virtual ~Agent(void) = default;
};

} /* namespace sparse_net_library */
#endif /* AGENT_H */

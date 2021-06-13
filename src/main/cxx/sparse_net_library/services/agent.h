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

#include <functional>

#include "gen/solution.pb.h"
#include "sparse_net_library/models/data_ringbuffer.h"
#include "sparse_net_library/models/data_pool.h"

namespace sparse_net_library{


using std::reference_wrapper;

/**
 * @brief      This class serves as a base for reinforcement learning agent, which provides output data
 *              based on different inputs
 */
class Agent{
public:
  Agent(uint32 required_temp_data_size_, uint32 required_tmp_buffer_num_)
  :  required_temp_data_size(required_temp_data_size_)
  ,  required_tmp_buffer_num(required_tmp_buffer_num_)
  { }

  /**
   * @brief      Solves the Solution provided in the constructor, previous neural information is supposedly available in @output buffer
   *
   * @param[in]      input    The input data to be taken
   * @param          output   The used Output data to write the results to
   */
  void solve(const vector<sdouble32>& input, DataRingbuffer& output) const;

  /**
   * @brief     Provids the size of the buffer it was declared with
   */
  uint32 get_required_temp_data_size(){
    return required_temp_data_size;
  }

  /**
   * @brief      Solves the Solution provided in the constructor, previous neural information is supposedly available in @output buffer
   *
   * @param[in]      input                    The input data to be taken
   * @param          output                   The used Output data to write the results to
   * @param[in]      tmp_data_pool            The already allocated data pool to be used to store intermediate data
   * @param[in]      used_data_pool_start     The first index inside @tmp_data_pool to be used
   */
  virtual void solve(
    const vector<sdouble32>& input, DataRingbuffer& output,
    const vector<reference_wrapper<vector<sdouble32>>>& tmp_data_pool,
    uint32 used_data_pool_start = 0
  ) const = 0;

  /**
   * @brief      Provide the underlying solution the solver is built to solve.
   *
   * @return     A const reference to the solution the solver is aiming to solve
   */
  virtual const Solution& get_solution(void) const = 0;

  virtual ~Agent(void) = default;
private:
  static DataPool<sdouble32> common_data_pool;

  uint32 required_temp_data_size;
  uint32 required_tmp_buffer_num;

};

} /* namespace sparse_net_library */
#endif /* AGENT_H */

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

#include "sparse_net_library/services/agent.h"

#include <vector>

namespace sparse_net_library{

using std::vector;

DataPool<sdouble32> Agent::common_data_pool;

void Agent::solve(const vector<sdouble32>& input, DataRingbuffer& output) const{
  vector<reference_wrapper<vector<sdouble32>>> tmp_data_pool;
  for(uint32 thread_iterator = 0; thread_iterator < required_tmp_buffer_num; ++thread_iterator)
    tmp_data_pool.push_back(common_data_pool.reserve_buffer(required_temp_data_size));

  solve(input, output, tmp_data_pool);

  for(vector<sdouble32>& tmp_buffer : tmp_data_pool) common_data_pool.release_buffer(tmp_buffer);
}

} /* namespace sparse_net_library */

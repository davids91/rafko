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

#ifndef SOLUTION_SOLVER_H
#define SOLUTION_SOLVER_H

#include "rafko_global.h"

#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <utility>
#include <functional>

#include "gen/solution.pb.h"
#include "rafko_gym/services/agent.h"
#include "rafko_utilities/models/data_ringbuffer.h"
#include "rafko_utilities/services/thread_group.h"

#include "sparse_net_library/services/partial_solution_solver.h"

namespace sparse_net_library{

using std::vector;
using std::mutex;
using std::reference_wrapper;
using std::unique_ptr;

using rafko_utilities::ThreadGroup;
using rafko_utilities::DataRingbuffer;
using rafko_gym::Agent;

/**
 * @brief      This class Processes a @Solution given in its constructor and handles
 *             the distribution of the needed resources for it.
 */
class Solution_solver : public Agent{
public:
  Solution_solver(const Solution_solver& other) = delete;/* Copy constructor */
  Solution_solver(Solution_solver&& other) = delete; /* Move constructor */
  Solution_solver& operator=(const Solution_solver& other) = delete; /* Copy assignment */
  Solution_solver& operator=(Solution_solver&& other) = delete; /* Move assignment */
  ~Solution_solver(void) = default;

  /* +++ Methods taken from @Agent +++ */
  void solve(
    const vector<sdouble32>& input, DataRingbuffer& output,
    const vector<reference_wrapper<vector<sdouble32>>>& tmp_data_pool, uint32 used_data_pool_start = 0
  ) const;
  const uint32 get_output_data_size(void) const{
    return solution.output_neuron_number();
  }
  using Agent::solve;
  /* --- Methods taken from @Agent --- */

private:
  Solution_solver(
    const Solution& to_solve, Service_context& context, vector<vector<Partial_solution_solver>> partial_solvers_,
    uint32 max_tmp_data_needed, uint32 max_tmp_data_needed_per_thread
  ): Agent(to_solve, max_tmp_data_needed, max_tmp_data_needed_per_thread, context.get_max_processing_threads())
  ,  solution(to_solve)
  ,  service_context(context)
  ,  partial_solvers(partial_solvers_)
  ,  execution_threads(context.get_max_solve_threads())
  { }

  const Solution& solution;
  Service_context& service_context;
  vector<vector<Partial_solution_solver>> partial_solvers;
  mutable mutex threads_mutex;
  ThreadGroup execution_threads;

public:
  class Builder{
  public:
    Builder(const Solution& to_solve, Service_context& context);
    unique_ptr<Solution_solver> build(void){
      return unique_ptr<Solution_solver>(new Solution_solver(
        solution, service_context, partial_solvers, max_tmp_size_needed, max_tmp_data_needed_per_thread
      ));
    }
  private:
    const Solution& solution;
    Service_context& service_context;
    vector<vector<Partial_solution_solver>> partial_solvers;
    uint32 max_tmp_size_needed = 0;
    uint32 max_tmp_data_needed_per_thread = 0;
  };
};

} /* namespace sparse_net_library */

#endif /* SOLUTION_SOLVER_H */

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
#include <thread>
#include <atomic>
#include <utility>
#include <functional>

#include "gen/solution.pb.h"
#include "sparse_net_library/services/agent.h"
#include "sparse_net_library/models/data_ringbuffer.h"

#include "sparse_net_library/services/partial_solution_solver.h"

namespace sparse_net_library{

using std::vector;
using std::thread;
using std::reference_wrapper;
using std::unique_ptr;

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

  /* +++ Methds taken from @Agent +++ */
  void solve(const vector<sdouble32>& input, DataRingbuffer& output, const vector<reference_wrapper<vector<sdouble32>>>& tmp_data_pool) const;
  const Solution& get_solution(void) const{
    return solution;
  }
  using Agent::solve;
  /* --- Methds taken from @Agent --- */

private:
  Solution_solver(const Solution& to_solve, Service_context& context, vector<vector<Partial_solution_solver>> partial_solvers_, uint32 max_tmp_data_needed)
  :  Agent(max_tmp_data_needed, context.get_max_solve_threads())
  ,  solution(to_solve)
  ,  service_context(context)
  ,  partial_solvers(partial_solvers_)
  { }

  const Solution& solution;
  Service_context& service_context;
  vector<vector<Partial_solution_solver>> partial_solvers;

public:
  class Builder{
  public:
    Builder(const Solution& to_solve, Service_context& context);
    unique_ptr<Solution_solver> build(void){
      return unique_ptr<Solution_solver>(new Solution_solver(solution, service_context, partial_solvers, max_tmp_data_needed));
    }
  private:
    const Solution& solution;
    Service_context& service_context;
    vector<vector<Partial_solution_solver>> partial_solvers;
    uint32 max_tmp_data_needed;
  };
};

} /* namespace sparse_net_library */

#endif /* SOLUTION_SOLVER_H */

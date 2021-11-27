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

#include "rafko_protocol/solution.pb.h"
#include "rafko_gym/services/rafko_agent.h"
#include "rafko_utilities/models/data_ringbuffer.h"
#include "rafko_utilities/services/thread_group.h"

#include "rafko_net/services/partial_solution_solver.h"

namespace rafko_net{

using std::vector;
using std::mutex;
using std::reference_wrapper;
using std::unique_ptr;

using rafko_utilities::ThreadGroup;
using rafko_utilities::DataRingbuffer;
using rafko_gym::RafkoAgent;

/**
 * @brief      This class Processes a @Solution given in its constructor and handles
 *             the distribution of the needed resources for it.
 */
class RAFKO_FULL_EXPORT SolutionSolver : public RafkoAgent{
public:
  SolutionSolver(const SolutionSolver& other) = delete;/* Copy constructor */
  SolutionSolver(SolutionSolver&& other) = delete; /* Move constructor */
  SolutionSolver& operator=(const SolutionSolver& other) = delete; /* Copy assignment */
  SolutionSolver& operator=(SolutionSolver&& other) = delete; /* Move assignment */
  ~SolutionSolver() = default;

  /* +++ Methods taken from @RafkoAgent +++ */
  void solve(
    const vector<sdouble32>& input, DataRingbuffer& output,
    const vector<reference_wrapper<vector<sdouble32>>>& tmp_data_pool, uint32 used_data_pool_start = 0
  ) const;
  uint32 get_output_data_size() const{
    return solution.output_neuron_number();
  }
  using RafkoAgent::solve;
  /* --- Methods taken from @RafkoAgent --- */

private:
  SolutionSolver(
    const Solution& to_solve, RafkoServiceContext& context, vector<vector<PartialSolution_solver>> partial_solvers_,
    uint32 max_tmp_data_needed, uint32 max_tmp_data_needed_per_thread
  ): RafkoAgent(to_solve, max_tmp_data_needed, max_tmp_data_needed_per_thread, context.get_max_processing_threads())
  ,  solution(to_solve)
  ,  service_context(context)
  ,  partial_solvers(partial_solvers_)
  ,  execution_threads(context.get_max_solve_threads())
  { }

  const Solution& solution;
  RafkoServiceContext& service_context;
  vector<vector<PartialSolution_solver>> partial_solvers;
  mutable mutex threads_mutex;
  ThreadGroup execution_threads;

public:
  class Builder{
  public:
    Builder(const Solution& to_solve, RafkoServiceContext& context);
    unique_ptr<SolutionSolver> build(){
      return unique_ptr<SolutionSolver>(new SolutionSolver(
        solution, service_context, partial_solvers, max_tmp_size_needed, max_tmp_data_needed_per_thread
      ));
    }
  private:
    const Solution& solution;
    RafkoServiceContext& service_context;
    vector<vector<PartialSolution_solver>> partial_solvers;
    uint32 max_tmp_size_needed = 0;
    uint32 max_tmp_data_needed_per_thread = 0;
  };
};

} /* namespace rafko_net */

#endif /* SOLUTION_SOLVER_H */

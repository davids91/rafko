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
#include <utility>
#include <functional>

#include "rafko_protocol/solution.pb.h"
#include "rafko_gym/models/rafko_agent.h"
#include "rafko_utilities/models/data_ringbuffer.h"
#include "rafko_utilities/services/thread_group.h"

#include "rafko_net/services/partial_solution_solver.h"
#include "rafko_net/services/rafko_net_feature_executor.h"

namespace rafko_net{

/**
 * @brief      This class Processes a @Solution given in its constructor and handles
 *             the distribution of the needed resources for it.
 */
class RAFKO_FULL_EXPORT SolutionSolver : public rafko_gym::RafkoAgent{
public:
  SolutionSolver(const SolutionSolver& other) = delete; /* Copy constructor */
  SolutionSolver(SolutionSolver&& other) = delete; /* Move constructor */
  SolutionSolver& operator=(const SolutionSolver& other) = delete; /* Copy assignment */
  SolutionSolver& operator=(SolutionSolver&& other) = delete; /* Move assignment */
  ~SolutionSolver() = default;

  /* +++ Methods taken from @RafkoAgent +++ */
  void solve(
    const std::vector<sdouble32>& input, rafko_utilities::DataRingbuffer& output,
    const std::vector<std::reference_wrapper<std::vector<sdouble32>>>& tmp_data_pool,
    uint32 used_data_pool_start = 0, uint32 thread_index = 0
  ) const;
  uint32 get_output_data_size() const{
    return solution.output_neuron_number();
  }
  using rafko_gym::RafkoAgent::solve;
  /* --- Methods taken from @RafkoAgent --- */

private:
  SolutionSolver(
    const Solution& to_solve, const rafko_mainframe::RafkoSettings& settings,
    std::vector<std::vector<PartialSolutionSolver>> partial_solvers_,
    uint32 max_tmp_data_needed, uint32 max_tmp_data_needed_per_thread
  );

  const rafko_mainframe::RafkoSettings& settings;
  std::vector<std::vector<PartialSolutionSolver>> partial_solvers;
  std::vector<std::unique_ptr<rafko_utilities::ThreadGroup>> execution_threads;
  RafkoNetFeatureExecutor feature_executor;

public:
  class Builder{
  public:
    Builder(const Solution& to_solve, const rafko_mainframe::RafkoSettings& settings);
    std::unique_ptr<rafko_net::SolutionSolver> build(){
      return std::unique_ptr<rafko_net::SolutionSolver>( new SolutionSolver(solution, settings, partial_solvers, max_tmp_size_needed, max_tmp_data_needed_per_thread) );
    }
  private:
    const Solution& solution;
    const rafko_mainframe::RafkoSettings& settings;
    std::vector<std::vector<PartialSolutionSolver>> partial_solvers;
    uint32 max_tmp_size_needed = 0u;
    uint32 max_tmp_data_needed_per_thread = 0u;
  };
};

} /* namespace rafko_net */

#endif /* SOLUTION_SOLVER_H */

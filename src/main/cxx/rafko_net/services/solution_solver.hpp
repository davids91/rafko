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

#include "rafko_global.hpp"

#include <vector>
#include <memory>
#include <atomic>
#include <utility>
#include <functional>

#include "rafko_protocol/solution.pb.h"
#include "rafko_gym/models/rafko_agent.hpp"
#include "rafko_utilities/models/data_ringbuffer.hpp"
#include "rafko_utilities/services/thread_group.hpp"

#include "rafko_net/services/partial_solution_solver.hpp"
#include "rafko_net/services/rafko_network_feature.hpp"
#include "rafko_gym/services/rafko_weight_adapter.hpp"

namespace rafko_net{

/**
 * @brief      This class Processes a @Solution given in its constructor and handles
 *             the distribution of the needed resources for it.
 */
class RAFKO_EXPORT SolutionSolver : public rafko_gym::RafkoAgent{
public:
  SolutionSolver(const Solution* to_solve, const rafko_mainframe::RafkoSettings& settings);

  SolutionSolver(const SolutionSolver& other) = delete; /* Copy constructor */
  SolutionSolver(SolutionSolver&& other) = delete; /* Move constructor */
  SolutionSolver& operator=(const SolutionSolver& other) = delete; /* Copy assignment */
  SolutionSolver& operator=(SolutionSolver&& other) = delete; /* Move assignment */
  ~SolutionSolver() = default;

  /**
   * @brief     Updates the stored @Solution pointer and rebuilds the underlying structure supporting it
   *
   * @param[in]     to_solve    The @SOlution pointer to rebuild the solver upon
   */
  void rebuild(const Solution* to_solve);

  /**
   * @brief     Exposes the feature executor used to run features on the network
   *
   * @return    A const reference of the exposed feature executor
   */
  constexpr const RafkoNetworkFeature& expose_executor(){
    return m_featureExecutor;
  }

  /* +++ Methods taken from @RafkoAgent +++ */
  rafko_utilities::ConstVectorSubrange<> solve(
    const std::vector<double>& input, bool reset_neuron_data = false, std::uint32_t thread_index = 0u
  ) override;

  void set_eval_mode(bool evaluation) override{
    evaluating = evaluation;
  }
  /* --- Methods taken from @RafkoAgent --- */

private:
  std::vector<std::vector<PartialSolutionSolver>> m_partialSolvers;
  std::vector<std::unique_ptr<rafko_utilities::ThreadGroup>> m_executionThreads;
  RafkoNetworkFeature m_featureExecutor;
  std::uint32_t m_maxTmpSizeNeeded = 0u;
  std::uint32_t m_maxTmpDataNeededPerThread = 0u;
  bool evaluating = true;

public:
  class RAFKO_EXPORT Factory{
  public:
    Factory(const RafkoNet& network, std::shared_ptr<const rafko_mainframe::RafkoSettings> settings);

    /**
     * @brief     Provides const access to the latest built solution
     *
     * @return    A const pointer to the last built Solution
     */
    constexpr const rafko_net::Solution* actual_solution(){
      return m_actualSolution;
    }

    /**
     * @brief     Provides const access to the used weight adapter so information might be queried based on it
     *
     * @return    A const reference to the used weight adapter
     */
    const rafko_gym::RafkoWeightAdapter& expose_weight_adapter(){
      return *m_weightAdapter;
    }

    /**
     * @brief     Updates the stored solution with the weights from the stored Neural Network reference
     */
    void refresh_actual_solution_weights(){
      m_weightAdapter->update_solution_with_weights();
    }

    /**
     * @brief     Builds a SolutionSolver and produces a pointer to it, based on its stored members
     *
     * param[in]    rebuild_solution    Creates a new @Solution object and stores it as reference
     *
     * @return    Ownership and pointer of the built solver
     */
    std::unique_ptr<SolutionSolver> build(bool rebuild_solution = false);

  private:
    const RafkoNet& m_network;
    std::shared_ptr<const rafko_mainframe::RafkoSettings> m_settings;
    rafko_net::Solution* m_actualSolution;
    std::unique_ptr<rafko_gym::RafkoWeightAdapter> m_weightAdapter;
    std::vector<std::unique_ptr<rafko_net::Solution>> m_ownedSolutions;
  }/*class SolutionSolver::Factory*/;
};

} /* namespace rafko_net */

#endif /* SOLUTION_SOLVER_H */

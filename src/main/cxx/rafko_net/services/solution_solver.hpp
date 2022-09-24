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
#include <mutex>
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
class RAFKO_EXPORT SolutionSolver : public virtual rafko_gym::RafkoAgent{
public:
  SolutionSolver(const Solution* to_solve, const rafko_mainframe::RafkoSettings& settings);

  SolutionSolver(const SolutionSolver& other) = delete; /* Copy constructor */
  SolutionSolver(SolutionSolver&& other) = delete; /* Move constructor */
  SolutionSolver& operator=(const SolutionSolver& other) = delete; /* Copy assignment */
  SolutionSolver& operator=(SolutionSolver&& other) = delete; /* Move assignment */
  ~SolutionSolver() = default;

  /**
   * @brief     Exposes the feature executor used to run features on the network
   *
   * @return    A const reference of the exposed feature executor
   */
  constexpr const RafkoNetworkFeature& expose_executor(){
    return m_featureExecutor;
  }

  /**
   * @brief      Provide the raw Neural data
   *
   * @param[in]      thread_index     The index of the target thread
   * @return         A const reference to the raw Neuron data
   */
  const rafko_utilities::DataRingbuffer<>& get_memory(std::uint32_t thread_index = 0) const{
    RFASSERT(thread_index < m_neuronValueBuffers.size());
    return m_neuronValueBuffers[thread_index];
  }

  /* +++ Methods taken from @RafkoAgent +++ */
  rafko_utilities::ConstVectorSubrange<> solve(
    const std::vector<double>& input, bool reset_neuron_data = false, std::uint32_t thread_index = 0u
  ) override;

  void set_eval_mode(bool evaluation) override{
    evaluating = evaluation;
  }
  #if(RAFKO_USES_OPENCL)
  /**
   * @brief     Sets the parameters the generated kernel code will be based on. These parameters usually come from an environment
   *
   * @param[in]     sequence_number                 How many sequences are there?
   * @param[in]     sequence_size                   How many feature-label pairs are in a sequence?
   * @param[in]     prefill_inputs_per_sequence     How many prefill inputs are there before a sequence?
   */
  constexpr void set_sequence_params(
    std::uint32_t sequence_number, std::uint32_t sequence_size = 1u, std::uint32_t prefill_inputs_per_sequence = 0u
  ){
    m_sequencesEvaluating = sequence_number;
    m_sequenceSize = sequence_size;
    m_prefillInputsPerSequence = prefill_inputs_per_sequence;
  }

  cl::Program::Sources get_step_sources() const override{
    return { rafko_net::SolutionBuilder::get_kernel_for_solution(
      *m_solution, "agent_solution",
      m_sequenceSize, m_prefillInputsPerSequence,
      m_settings
    ) };
  }

  std::vector<std::string> get_step_names() const override{
    return {"agent_solution"};
  }

  /**
   * @brief      Provides the input dimension of the agent, which consist of
   *             3 buffers: mode, weights, and (inputs + prefill) for each evaluated sequence
   *
   * @return     Vector of dimensions in order of @get_step_sources and @get_step_names
   */
  std::vector<rafko_mainframe::RafkoNBufShape> get_input_shapes() const override{
    return{ rafko_mainframe::RafkoNBufShape{
      1u, m_deviceWeightTableSize,
      (m_sequencesEvaluating * (m_sequenceSize + m_prefillInputsPerSequence) * m_solution->network_input_size())
    } };
  }

  /**
   * @brief      Provides the output dimension of the agent, which consist of
   *             1 buffer: Neuron outputs for each evaluated sequence or network memory
   *
   * @return     Vector of dimensions in order of @get_step_sources and @get_step_names
   *             Agent output structure: {{ used bytes for execution, used bytes for performance feature error summary }}
   */
  std::vector<rafko_mainframe::RafkoNBufShape> get_output_shapes() const override{
    const std::size_t bytes_used = (
      std::max(m_sequencesEvaluating, 1u) /* number of sequences to evaluate */
      * std::max(2u, std::max( /* number of labels per sequence */
        m_solution->network_memory_length(), (m_sequenceSize + m_prefillInputsPerSequence)
      ) )
      * m_solution->neuron_number() /* number of numbers per label */
    );
    return{ rafko_mainframe::RafkoNBufShape{bytes_used, 1u} };
  }

  std::tuple<cl::NDRange,cl::NDRange,cl::NDRange> get_solution_space() const override{
    return {cl::NullRange,cl::NDRange(m_sequencesEvaluating),cl::NullRange};
  }
  #endif/*(RAFKO_USES_OPENCL)*/
  /* --- Methods taken from @RafkoAgent --- */

private:
  const rafko_net::Solution* m_solution;
  std::uint32_t m_maxThreadNumber;
  rafko_utilities::DataPool<double> m_commonDataPool;
  std::vector<rafko_utilities::DataRingbuffer<>> m_neuronValueBuffers; /* One rafko_utilities::DataRingbuffer per thread */
  std::vector<std::reference_wrapper<std::vector<double>>> m_usedDataBuffers;
  std::vector<std::vector<PartialSolutionSolver>> m_partialSolvers;
  std::vector<std::unique_ptr<rafko_utilities::ThreadGroup>> m_executionThreads;
  RafkoNetworkFeature m_featureExecutor;
  std::uint32_t m_maxTmpSizeNeeded = 0u;
  std::uint32_t m_maxTmpDataNeededPerThread = 0u;
  std::mutex m_structureMutex;
  bool evaluating = true;

  #if(RAFKO_USES_OPENCL)
    std::uint32_t m_sequencesEvaluating = 1u;
    std::uint32_t m_sequenceSize = 1u;
    std::uint32_t m_prefillInputsPerSequence = 0u;
    std::uint32_t m_deviceWeightTableSize;
  #endif/*(RAFKO_USES_OPENCL)*/

  /**
   * @brief     Updates the stored @Solution pointer and rebuilds the underlying structure supporting it
   *
   * @param[in]     to_solve    The @SOlution pointer to rebuild the solver upon
   */
  void rebuild(const Solution* to_solve);

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
     * param[in]    swap_solution       When true, no new Solution is stored, instead the last built
     *                                  solution is swapped with the newly built one.
     *
     * @return    Ownership and pointer of the built solver
     */
    std::shared_ptr<SolutionSolver> build(bool rebuild_solution = false, bool swap_solution = false);

  private:
    const RafkoNet& m_network;
    std::shared_ptr<const rafko_mainframe::RafkoSettings> m_settings;
    rafko_net::Solution* m_actualSolution;
    std::unique_ptr<rafko_gym::RafkoWeightAdapter> m_weightAdapter;
    std::vector<std::unique_ptr<rafko_net::Solution>> m_ownedSolutions;
    std::vector<std::shared_ptr<rafko_net::SolutionSolver>> m_ownedSolvers;
  }/*class SolutionSolver::Factory*/;
};

} /* namespace rafko_net */

#endif /* SOLUTION_SOLVER_H */

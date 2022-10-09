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

#include "rafko_net/services/solution_solver.hpp"

#include <stdexcept>
#include <mutex>

#include "rafko_net/models/neuron_info.hpp"
#include "rafko_net/services/synapse_iterator.hpp"
#include "rafko_net/services/solution_builder.hpp"
#include "rafko_net/services/rafko_network_feature.hpp"

namespace rafko_net{

SolutionSolver::Factory::Factory(const RafkoNet& network, std::shared_ptr<const rafko_mainframe::RafkoSettings> settings)
: m_network(network)
, m_settings(settings)
, m_actualSolution(SolutionBuilder(*m_settings).build(m_network))
, m_weightAdapter(std::make_unique<rafko_gym::RafkoWeightAdapter>(m_network, *m_actualSolution, *m_settings))
{
  if(nullptr == m_settings->get_arena_ptr()){
    m_ownedSolutions.emplace_back(m_actualSolution);
  }
}

std::shared_ptr<SolutionSolver> SolutionSolver::Factory::build(bool rebuild_solution, bool swap_solution){
  if(rebuild_solution){
    if(swap_solution){
      SolutionBuilder(*m_settings).update(m_actualSolution, m_network);
      for(std::shared_ptr<SolutionSolver>& solver : m_ownedSolvers)
        solver->rebuild(m_actualSolution);
    }else{
      m_actualSolution = SolutionBuilder(*m_settings).build(m_network);
      m_ownedSolvers.clear(); /* A new Solution object is built, so previously built solvers are not handled within this factory anymore */
      if(nullptr == m_settings->get_arena_ptr()){
        m_ownedSolutions.emplace_back(m_actualSolution);
      }
    }
    m_weightAdapter = std::make_unique<rafko_gym::RafkoWeightAdapter>(m_network, *m_actualSolution, *m_settings);
  }else if(swap_solution)
    throw std::runtime_error("Error: Nothing to swap the actual Solution with!");

  RFASSERT(static_cast<bool>(m_actualSolution));
  m_ownedSolvers.push_back(std::make_shared<SolutionSolver>(m_actualSolution, *m_settings));
  return m_ownedSolvers.back();
}

SolutionSolver::SolutionSolver(const Solution* to_solve, const rafko_mainframe::RafkoSettings& settings)
: rafko_gym::RafkoAgent(settings)
, m_solution(to_solve)
, m_maxThreadNumber(settings.get_max_processing_threads())
, m_featureExecutor(m_executionThreads)
#if(RAFKO_USES_OPENCL)
, m_deviceWeightTableSize( std::accumulate(
  m_solution->partial_solutions().begin(), m_solution->partial_solutions().end(), 0u,
  [](const std::uint32_t& sum, const rafko_net::PartialSolution& partial){
    return ( sum + partial.weight_table_size() );
  }
) )
#endif/*(RAFKO_USES_OPENCL)*/
{
  RFASSERT(m_solution);
  rebuild(m_solution);
  for(std::uint32_t thread_index = 0; thread_index < m_settings.get_max_processing_threads(); ++ thread_index)
    m_executionThreads.emplace_back(std::make_unique<rafko_utilities::ThreadGroup>(settings.get_max_solve_threads()));
}

void SolutionSolver::rebuild(const Solution* to_solve){
  std::lock_guard<std::mutex> my_lock(m_structureMutex);
  m_maxTmpSizeNeeded = 0u;
  m_maxTmpDataNeededPerThread = 0u;
  m_partialSolvers.clear();
  m_solution = to_solve;
  m_neuronValueBuffers.clear();

  std::uint32_t partial_index_at_row_start = 0u;
  for(std::int32_t row_iterator = 0; row_iterator < m_solution->cols_size(); ++row_iterator){
    m_partialSolvers.push_back(std::vector<PartialSolutionSolver>());
    for(std::uint32_t column_index = 0; column_index < m_solution->cols(row_iterator); ++column_index){
      m_partialSolvers[row_iterator].emplace_back(
        m_solution->partial_solutions(partial_index_at_row_start + column_index), m_settings
      ); /* Initialize a solver for this partial solution element */
      if(m_partialSolvers[row_iterator][column_index].get_required_tmp_data_size() > m_maxTmpSizeNeeded)
        m_maxTmpSizeNeeded = m_partialSolvers[row_iterator][column_index].get_required_tmp_data_size();
    }
    partial_index_at_row_start += m_solution->cols(row_iterator);
    if(m_solution->cols(row_iterator) > m_maxTmpDataNeededPerThread)
      m_maxTmpDataNeededPerThread = m_solution->cols(row_iterator);
  } /* loop through every partial solution and initialize solvers and output maps for them */

  /* Actualize buffers and buffer sizes: A temporary buffer is allocated for future usage per thread */
  while(m_usedDataBuffers.size() < (m_maxThreadNumber * m_maxTmpDataNeededPerThread))
    m_usedDataBuffers.push_back(m_commonDataPool.reserve_buffer(m_maxTmpSizeNeeded));
  for(std::vector<double>& buffer : m_usedDataBuffers)buffer.resize(m_maxTmpSizeNeeded);
  for(std::uint32_t thread_index = 0; thread_index < m_maxThreadNumber; ++ thread_index){
    m_neuronValueBuffers.emplace_back( m_solution->network_memory_length(),
      [this](std::vector<double>& buffer){
        buffer = std::vector<double>(m_solution->neuron_number(), 0.0);
      }
    );
  }
}

rafko_utilities::ConstVectorSubrange<> SolutionSolver::solve(
  const std::vector<double>& input, bool reset_neuron_data, std::uint32_t thread_index
){
  if(m_maxThreadNumber > thread_index){
    if( input.size() != m_solution->network_input_size() )
      throw std::runtime_error("Input size(" + std::to_string(input.size()) + ") doesn't match "
        + std::string("networks input size(") + std::to_string(m_solution->network_input_size()) + ")!"
      );

    if(reset_neuron_data)m_neuronValueBuffers[thread_index].reset();
    const std::uint32_t used_data_pool_start = thread_index * m_maxTmpDataNeededPerThread;
    if(0 < m_solution->cols_size()){
      std::uint32_t partial_index = 0;
      std::uint32_t col_iterator;
      std::mutex solved_features_mutex;
      std::vector<std::reference_wrapper<const FeatureGroup>> solved_features;

      m_neuronValueBuffers[thread_index].copy_step(); /* move the iterator forward to the next slot and store the current data */
      for(std::int32_t row_iterator = 0; row_iterator < m_solution->cols_size(); ++row_iterator){
        if(0 == m_solution->cols(row_iterator)) throw std::runtime_error("A solution row of 0 columns!");
        col_iterator = 0;
        if( /* Don't use the threadgroup if there is no need for multiple threads.. */
          (m_solution->cols(row_iterator) < m_settings.get_max_solve_threads()/2u)
          ||(m_solution->cols(row_iterator) < 2u) /* ..since the number of partial solutions depend on the available device size */
        ){ /* having fewer partial solutions in a row usually implies whether or not multiple threads are needed */
          while(col_iterator < m_solution->cols(row_iterator)){
            for(std::uint16_t inner_thread_index = 0; inner_thread_index < m_settings.get_max_solve_threads(); ++inner_thread_index){
              if(col_iterator < m_solution->cols(row_iterator)){
                m_partialSolvers[row_iterator][col_iterator].solve(
                  std::ref(input), std::ref(m_neuronValueBuffers[thread_index]), std::ref(m_usedDataBuffers[used_data_pool_start + inner_thread_index].get())
                );
                const PartialSolution& partial = m_solution->partial_solutions(partial_index);
                for(std::int32_t feature_index = 0; feature_index < partial.solved_features_size(); feature_index++)
                  solved_features.push_back( partial.solved_features(feature_index) );
                ++col_iterator;
                ++partial_index;
              }else break;
            }
          }/* while(col_iterator < solution.cols(row_iterator)) */
        }else{
          while(col_iterator < m_solution->cols(row_iterator)){
            { /* To make the Solver itself thread-safe; the sub-threads need to be guarded with a lock */
              m_executionThreads[thread_index]->start_and_block(
              [this, &input, used_data_pool_start, row_iterator, col_iterator, partial_index, &solved_features_mutex, &solved_features, thread_index](std::uint32_t inner_thread_index){
                if((col_iterator + inner_thread_index) < m_solution->cols(row_iterator)){
                  m_partialSolvers[row_iterator][(col_iterator + inner_thread_index)].solve(
                    input, m_neuronValueBuffers[thread_index], m_usedDataBuffers[used_data_pool_start + inner_thread_index].get()
                  );
                  const PartialSolution& partial = m_solution->partial_solutions(partial_index + inner_thread_index);
                  for(std::int32_t feature_index = 0; feature_index < partial.solved_features_size(); feature_index++){
                    std::lock_guard<std::mutex> my_lock(solved_features_mutex);
                    solved_features.push_back( partial.solved_features(feature_index) );
                    /*!Note: multiple features might be solved at the same time, but theoretically they shouldn't clash
                     * because of the Neuron router filtering.
                     */
                  }
                }
              });
            }
            partial_index += m_settings.get_max_solve_threads();
            col_iterator += m_settings.get_max_solve_threads();
          } /* while(col_iterator < solution.cols(row_iterator)) */
        }
        /*!Note: Triggered feature groups are only solved after the row for consistency, since columns inside rows are solved in paralell,
         * and each column may contain feature relevant to any @Neuron inside the the current row.
         */
        for(std::uint32_t feature_index = 0; feature_index < solved_features.size(); feature_index++){
          if((m_evaluating) || NeuronInfo::is_feature_relevant_to_solution( solved_features[feature_index].get().feature() )){
            /*!Note: training relevant features only need to be run during evaluation */
            m_featureExecutor.execute_solution_relevant(
              solved_features[feature_index], m_settings,
              {m_neuronValueBuffers[thread_index].get_element(0u)}, thread_index
            );
          }
        }
        solved_features.clear();
      } /* for(every row in the @Solution) */

      return { /* return with the range of the output Neurons */
        m_neuronValueBuffers[thread_index].get_element(0).end() - m_solution->output_neuron_number(),
        m_neuronValueBuffers[thread_index].get_element(0).end()
      };
    }else throw std::runtime_error("A solution of 0 rows!");
  }else throw std::runtime_error("Thread index out of bounds!");

}

} /* namespace rafko_net */

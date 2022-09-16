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

SolutionSolver::Builder::Builder(const Solution* to_solve, const rafko_mainframe::RafkoSettings& settings)
:  m_solution(to_solve)
,  m_settings(settings)
{
  std::uint32_t partial_index_at_row_start = 0;
  for(std::int32_t row_iterator = 0; row_iterator < m_solution->cols_size(); ++row_iterator){
    m_partialSolvers.push_back(std::vector<PartialSolutionSolver>());
    for(std::uint32_t column_index = 0; column_index < m_solution->cols(row_iterator); ++column_index){
      m_partialSolvers[row_iterator].push_back( PartialSolutionSolver(
        m_solution->partial_solutions(partial_index_at_row_start + column_index), settings
      )); /* Initialize a solver for this partial solution element */
      if(m_partialSolvers[row_iterator][column_index].get_required_tmp_data_size() > m_maxTmpSizeNeeded)
        m_maxTmpSizeNeeded = m_partialSolvers[row_iterator][column_index].get_required_tmp_data_size();
    }
    partial_index_at_row_start += m_solution->cols(row_iterator);
    if(m_solution->cols(row_iterator) > m_maxTmpDataNeededPerThread)
      m_maxTmpDataNeededPerThread = m_solution->cols(row_iterator);
  } /* loop through every partial solution and initialize solvers and output maps for them */
}

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

std::unique_ptr<SolutionSolver> SolutionSolver::Factory::build(bool rebuild_solution){
  if(rebuild_solution){
    m_actualSolution = SolutionBuilder(*m_settings).build(m_network);
    if(nullptr == m_settings->get_arena_ptr()){
      m_ownedSolutions.emplace_back(m_actualSolution);
    }
    m_weightAdapter = std::make_unique<rafko_gym::RafkoWeightAdapter>(m_network, *m_actualSolution, *m_settings);
  }

  RFASSERT(static_cast<bool>(m_actualSolution));
  return Builder(m_actualSolution, *m_settings).build();
}

SolutionSolver::SolutionSolver(
  const Solution* to_solve, const rafko_mainframe::RafkoSettings& settings,
  std::vector<std::vector<PartialSolutionSolver>> partial_solvers,
  std::uint32_t max_tmp_data_needed, std::uint32_t max_tmp_data_needed_per_thread
): rafko_gym::RafkoAgent(to_solve, settings, max_tmp_data_needed, max_tmp_data_needed_per_thread, settings.get_max_processing_threads())
,  m_partialSolvers(partial_solvers)
,  m_featureExecutor(m_executionThreads)
{
  for(std::uint32_t thread_index = 0; thread_index < m_settings.get_max_processing_threads(); ++ thread_index)
    m_executionThreads.emplace_back(std::make_unique<rafko_utilities::ThreadGroup>(settings.get_max_solve_threads()));
}

void SolutionSolver::solve(
  const std::vector<double>& input, rafko_utilities::DataRingbuffer<>& output,
  const std::vector<std::reference_wrapper<std::vector<double>>>& tmp_data_pool,
  std::uint32_t used_data_pool_start, std::uint32_t thread_index
) const{
  if(0 < m_solution->cols_size()){
    std::uint32_t col_iterator;
    std::mutex solved_features_mutex;
    std::vector<std::reference_wrapper<const FeatureGroup>> solved_features;

    output.copy_step(); /* move the iterator forward to the next slot and store the current data */
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
                std::ref(input), std::ref(output), std::ref(tmp_data_pool[used_data_pool_start + inner_thread_index].get())
              );
              const PartialSolution& partial = m_partialSolvers[row_iterator][col_iterator].get_partial();
              for(std::int32_t feature_index = 0; feature_index < partial.solved_features_size(); feature_index++)
                solved_features.push_back( partial.solved_features(feature_index) );
              ++col_iterator;
            }else break;
          }
        }/* while(col_iterator < solution.cols(row_iterator)) */
      }else{
        while(col_iterator < m_solution->cols(row_iterator)){
          { /* To make the Solver itself thread-safe; the sub-threads need to be guarded with a lock */
            m_executionThreads[thread_index]->start_and_block(
            [this, &input, &output, &tmp_data_pool, used_data_pool_start, row_iterator, col_iterator, &solved_features_mutex, &solved_features](std::uint32_t inner_thread_index){
              if((col_iterator + inner_thread_index) < m_solution->cols(row_iterator)){
                m_partialSolvers[row_iterator][(col_iterator + inner_thread_index)].solve(
                  input,output,tmp_data_pool[used_data_pool_start + inner_thread_index].get()
                );
                const PartialSolution& partial = m_partialSolvers[row_iterator][col_iterator].get_partial();
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
          col_iterator += m_settings.get_max_solve_threads();
        } /* while(col_iterator < solution.cols(row_iterator)) */
      }
      /*!Note: Triggered feature groups are only solved after the row for consistency, since columns inside rows are solved in paralell,
       * and each column may contain feature relevant to any @Neuron inside the the current row.
       */
      for(std::uint32_t feature_index = 0; feature_index < solved_features.size(); feature_index++){
        if(
          (evaluating)
          || NeuronInfo::is_feature_relevant_to_training( solved_features[feature_index].get().feature() )
        ){
          /*!Note: training relevant features only need to be run during evaluation */
          m_featureExecutor.execute_solution_relevant(
            solved_features[feature_index], m_settings,
            {output.get_element(0u)}, thread_index
          );
        }
      }
      solved_features.clear();
    } /* for(every row in the @Solution) */
  }else throw std::runtime_error("A solution of 0 rows!");
}

} /* namespace rafko_net */

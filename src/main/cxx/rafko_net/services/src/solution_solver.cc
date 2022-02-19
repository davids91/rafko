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

#include "rafko_net/services/solution_solver.h"

#include <stdexcept>
#include <mutex>

#include "rafko_net/models/neuron_info.h"
#include "rafko_net/services/synapse_iterator.h"
#include "rafko_net/services/rafko_network_feature.h"

namespace rafko_net{

SolutionSolver::Builder::Builder(const Solution& to_solve, const rafko_mainframe::RafkoSettings& settings)
:  solution(to_solve)
,  settings(settings)
{
  uint32 partial_index_at_row_start = 0;
  for(sint32 row_iterator = 0; row_iterator < solution.cols_size(); ++row_iterator){
    partial_solvers.push_back(std::vector<PartialSolutionSolver>());
    for(uint32 column_index = 0; column_index < solution.cols(row_iterator); ++column_index){
      partial_solvers[row_iterator].push_back( PartialSolutionSolver(
        solution.partial_solutions(partial_index_at_row_start + column_index), settings
      )); /* Initialize a solver for this partial solution element */
      if(partial_solvers[row_iterator][column_index].get_required_tmp_data_size() > max_tmp_size_needed)
        max_tmp_size_needed = partial_solvers[row_iterator][column_index].get_required_tmp_data_size();
    }
    partial_index_at_row_start += solution.cols(row_iterator);
    if(solution.cols(row_iterator) > max_tmp_data_needed_per_thread)
      max_tmp_data_needed_per_thread = solution.cols(row_iterator);
  } /* loop through every partial solution and initialize solvers and output maps for them */
}

SolutionSolver::SolutionSolver(
  const Solution& to_solve, const rafko_mainframe::RafkoSettings& settings_,
  std::vector<std::vector<PartialSolutionSolver>> partial_solvers_,
  uint32 max_tmp_data_needed, uint32 max_tmp_data_needed_per_thread
): rafko_gym::RafkoAgent(to_solve, settings_, max_tmp_data_needed, max_tmp_data_needed_per_thread, settings_.get_max_processing_threads())
,  settings(settings_)
,  partial_solvers(partial_solvers_)
,  feature_executor(execution_threads)
{
  for(uint32 thread_index = 0; thread_index < settings.get_max_processing_threads(); ++ thread_index)
    execution_threads.push_back(std::make_unique<rafko_utilities::ThreadGroup>(settings.get_max_solve_threads()));
}

void SolutionSolver::solve(
  const std::vector<sdouble32>& input, rafko_utilities::DataRingbuffer& output,
  const std::vector<std::reference_wrapper<std::vector<sdouble32>>>& tmp_data_pool,
  uint32 used_data_pool_start, uint32 thread_index
) const{
  if(0 < solution.cols_size()){
    uint32 col_iterator;
    std::mutex solved_features_mutex;
    std::vector<std::reference_wrapper<const FeatureGroup>> solved_features;

    output.step(); /* move the iterator forward to the next slot and store the current data */
    for(sint32 row_iterator = 0; row_iterator < solution.cols_size(); ++row_iterator){
      if(0 == solution.cols(row_iterator)) throw std::runtime_error("A solution row of 0 columns!");
      col_iterator = 0;
      if( /* Don't use the threadgroup if there is no need for multiple threads.. */
        (solution.cols(row_iterator) < settings.get_max_solve_threads()/2u)
        ||(solution.cols(row_iterator) < 2u) /* ..since the number of partial solutions depend on the available device size */
      ){ /* having fewer partial solutions in a row usually implies whether or not multiple threads are needed */
        while(col_iterator < solution.cols(row_iterator)){
          for(uint16 inner_thread_index = 0; inner_thread_index < settings.get_max_solve_threads(); ++inner_thread_index){
            if(col_iterator < solution.cols(row_iterator)){
              partial_solvers[row_iterator][col_iterator].solve(
                std::ref(input), std::ref(output), std::ref(tmp_data_pool[used_data_pool_start + inner_thread_index].get())
              );
              const PartialSolution& partial = partial_solvers[row_iterator][col_iterator].get_partial();
              for(sint32 feature_index = 0; feature_index < partial.solved_features_size(); feature_index++)
                solved_features.push_back( partial.solved_features(feature_index) );
              ++col_iterator;
            }else break;
          }
        }/* while(col_iterator < solution.cols(row_iterator)) */
      }else{
        while(col_iterator < solution.cols(row_iterator)){
          { /* To make the Solver itself thread-safe; the sub-threads need to be guarded with a lock */
            execution_threads[thread_index]->start_and_block(
            [this, &input, &output, &tmp_data_pool, used_data_pool_start, row_iterator, col_iterator, &solved_features_mutex, &solved_features](uint32 inner_thread_index){
              if((col_iterator + inner_thread_index) < solution.cols(row_iterator)){
                partial_solvers[row_iterator][(col_iterator + inner_thread_index)].solve(
                  input,output,tmp_data_pool[used_data_pool_start + inner_thread_index].get()
                );
                const PartialSolution& partial = partial_solvers[row_iterator][col_iterator].get_partial();
                for(sint32 feature_index = 0; feature_index < partial.solved_features_size(); feature_index++){
                  std::lock_guard<std::mutex> my_lock(solved_features_mutex);
                  solved_features.push_back( partial.solved_features(feature_index) );
                  /*!Note: multiple features might be solved at the same time, but theoretically they shouldn't clash
                   * because of the Neuron router filtering.
                   */
                }
              }
            });
          }
          col_iterator += settings.get_max_solve_threads();
        } /* while(col_iterator < solution.cols(row_iterator)) */
      }
      /*!Note: Triggered feature groups are only solved after the row for consistency, since columns inside rows are solved in paralell,
       * and each column may contain feature relevant to any @Neuron inside the the current row.
       */
      for(uint32 feature_index = 0; feature_index < solved_features.size(); feature_index++){
        if(
          (evaluating)
          || NeuronInfo::is_feature_relevant_to_training( solved_features[feature_index].get().feature() )
        ){
          /*!Note: training relevant features only need to be run during evaluation */
          feature_executor.execute_solution_relevant(
            solved_features[feature_index], settings, output.get_element(0u),
            thread_index
          );
        }
      }
      solved_features.clear();
    } /* for(every row in the @Solution) */
  }else throw std::runtime_error("A solution of 0 rows!");
}

} /* namespace rafko_net */

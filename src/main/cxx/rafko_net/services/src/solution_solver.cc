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

#include "rafko_net/services/synapse_iterator.h"

namespace rafko_net{

using std::function;
using std::ref;

Solution_solver::Builder::Builder(const Solution& to_solve, Service_context& context)
:  solution(to_solve)
,  service_context(context)
{
  uint32 partial_index_at_row_start = 0;
  for(sint32 row_iterator = 0; row_iterator < solution.cols_size(); ++row_iterator){
    partial_solvers.push_back(vector<Partial_solution_solver>());
    for(uint32 column_index = 0; column_index < solution.cols(row_iterator); ++column_index){
      partial_solvers[row_iterator].push_back( Partial_solution_solver(
        solution.partial_solutions(partial_index_at_row_start + column_index), context
      )); /* Initialize a solver for this partial solution element */
      if(partial_solvers[row_iterator][column_index].get_required_tmp_data_size() > max_tmp_size_needed)
        max_tmp_size_needed = partial_solvers[row_iterator][column_index].get_required_tmp_data_size();
    }
    partial_index_at_row_start += solution.cols(row_iterator);
    if(solution.cols(row_iterator) > max_tmp_data_needed_per_thread)
      max_tmp_data_needed_per_thread = solution.cols(row_iterator);
  } /* loop through every partial solution and initialize solvers and output maps for them */
}

void Solution_solver::solve(
  const vector<sdouble32>& input, DataRingbuffer& output,
  const vector<reference_wrapper<vector<sdouble32>>>& tmp_data_pool, uint32 used_data_pool_start
) const{
  if(0 < solution.cols_size()){
    uint32 col_iterator;
    output.step(); /* move the iterator forward for the next one and store the current data */
    for(sint32 row_iterator = 0; row_iterator < solution.cols_size(); ++row_iterator){
      if(0 == solution.cols(row_iterator)) throw std::runtime_error("A solution row of 0 columns!");
      col_iterator = 0;
      if( /* Don't use the threadgroup if there is no need for multiple threads.. */
        (solution.cols(row_iterator) < service_context.get_max_solve_threads()/2u)
        ||(solution.cols(row_iterator) < 2u) /* ..since the number of partial solutions depend on the available device size */
      ){ /* having fewer partial solutions in a row usually implies whether or not multiple threads are needed */
        while(col_iterator < solution.cols(row_iterator)){
          for(uint16 thread_index = 0; thread_index < service_context.get_max_solve_threads(); ++thread_index){
            if(col_iterator < solution.cols(row_iterator)){
              partial_solvers[row_iterator][col_iterator].solve(ref(input), ref(output), ref(tmp_data_pool[used_data_pool_start + thread_index].get()));
              ++col_iterator;
            }else break;
          }
        }/* while(col_iterator < solution.cols(row_iterator)) */
      }else{
        while(col_iterator < solution.cols(row_iterator)){
          { /* To make the Solver itself thread-safe; the sub-threads need to be guarded with a lock */
            std::lock_guard<mutex> my_lock(threads_mutex);
            execution_threads.start_and_block([this, &input, &output, &tmp_data_pool, used_data_pool_start, row_iterator, col_iterator]
            (uint32 thread_index){
              if((col_iterator + thread_index) < solution.cols(row_iterator)){
                partial_solvers[row_iterator][(col_iterator + thread_index)].solve(
                  input,output,tmp_data_pool[used_data_pool_start + thread_index].get()
                );
              }
            });
          }
          col_iterator += service_context.get_max_solve_threads();
        } /* while(col_iterator < solution.cols(row_iterator)) */
      }
    } /* for(sint32 row_iterator = 0; row_iterator < solution.cols_size(); ++row_iterator) */
  }else throw std::runtime_error("A solution of 0 rows!");
}

} /* namespace rafko_net */

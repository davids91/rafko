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

#include "sparse_net_library/services/solution_solver.h"

#include <thread>
#include <stdexcept>

#include "sparse_net_library/services/synapse_iterator.h"

namespace sparse_net_library{

using std::ref;
using std::thread;

Solution_solver::Builder::Builder(const Solution& to_solve, Service_context& context)
:  solution(to_solve)
,  service_context(context)
,  max_tmp_data_needed(0u)
{
  partial_solvers = vector<vector<Partial_solution_solver>>();
  uint32 partial_index_at_row_start = 0;
  for(sint32 row_iterator = 0; row_iterator < solution.cols_size(); ++row_iterator){
    partial_solvers.push_back(vector<Partial_solution_solver>());
    for(uint32 column_index = 0; column_index < solution.cols(row_iterator); ++column_index){
      partial_solvers[row_iterator].push_back( Partial_solution_solver(
        solution.partial_solutions(partial_index_at_row_start + column_index), context
      )); /* Initialize a solver for this partial solution element */
      if(partial_solvers[row_iterator][column_index].get_required_tmp_data_size() > max_tmp_data_needed)
        max_tmp_data_needed = partial_solvers[row_iterator][column_index].get_required_tmp_data_size();
    }
    partial_index_at_row_start += solution.cols(row_iterator);
  } /* loop through every partial solution and initialize solvers and output maps for them */
}

void Solution_solver::solve(
  const vector<sdouble32>& input, DataRingbuffer& output,
  const vector<reference_wrapper<vector<sdouble32>>>& tmp_data_pool,
  uint32 used_data_pool_start
) const{
  if(0 < solution.cols_size()){
    uint32 col_iterator;
    vector<thread> solve_threads;
    output.step(); /* move the iterator forward for the next one and store the current data */
    for(sint32 row_iterator = 0; row_iterator < solution.cols_size(); ++row_iterator){
      col_iterator = 0;
      if(0 == solution.cols(row_iterator)) throw std::runtime_error("A solution row of 0 columns!");
      while(col_iterator < solution.cols(row_iterator)){
        for(uint16 thread_index = 0; thread_index < service_context.get_max_solve_threads(); ++thread_index){
          if(col_iterator < solution.cols(row_iterator)){
            void (Partial_solution_solver::*solve_func_ptr)(const vector<sdouble32>&, DataRingbuffer&, vector<sdouble32>&) const
              = &Partial_solution_solver::solve;
            solve_threads.push_back(thread(
              solve_func_ptr, partial_solvers[row_iterator][col_iterator],
              ref(input), ref(output), ref(tmp_data_pool[used_data_pool_start + thread_index].get())
            ));
            ++col_iterator;
          }else break;
        }
        std::for_each(solve_threads.begin(),solve_threads.end(),[](thread& solve_thread){
          if(true == solve_thread.joinable())solve_thread.join();
        });
        solve_threads.clear();
      }
    } /* for(sint32 row_iterator = 0; row_iterator < solution.cols_size(); ++row_iterator) */
  }else throw std::runtime_error("A solution of 0 rows!");
}

} /* namespace sparse_net_library */

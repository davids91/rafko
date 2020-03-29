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

#include "services/solution_solver.h"
#include "services/synapse_iterator.h"

#include <cmath>

namespace sparse_net_library{

using std::swap_ranges;

Solution_solver::Solution_solver(const Solution& to_solve, Service_context context)
:  solution(to_solve)
,  neuron_data(std::min(1u,solution.network_memory_length()), solution.neuron_number())
,  transfer_function_input(solution.neuron_number(),double_literal(0.0))
,  transfer_function_output(solution.neuron_number(),double_literal(0.0))
,  number_of_threads(context.get_max_solve_threads())
{
  partial_solvers = vector<vector<Partial_solution_solver>>();
  for(sint32 row_iterator = 0; row_iterator < solution.cols_size(); ++row_iterator){
    partial_solvers.push_back(vector<Partial_solution_solver>());
    for(uint32 column_index = 0; column_index < solution.cols(row_iterator); ++column_index){
      partial_solvers[row_iterator].push_back( Partial_solution_solver(
        get_partial(row_iterator,column_index,solution), neuron_data
      )); /* Initialize a solver for this partial solution element */
    }
  } /* loop through every partial solution and initialize solvers and output maps for them */
}

void Solution_solver::solve(const vector<sdouble32>& input){

  using std::ref;
  using std::min;

  if(0 < solution.cols_size()){
    vector<thread> solve_threads;
    uint32 col_iterator;
    for(sint32 row_iterator = 0; row_iterator < solution.cols_size(); ++row_iterator){
      col_iterator = 0;
      if(0 == solution.cols(row_iterator)) throw "A solution row of 0 columns!";
      while(col_iterator < solution.cols(row_iterator)){
        for(uint16 i = 0; i < number_of_threads; ++i){
          if(col_iterator < solution.cols(row_iterator)){
            solve_threads.push_back(thread(&Solution_solver::solve_a_partial,
              this, ref(input), row_iterator, col_iterator
            ));
            ++col_iterator;
          }else break;
        }
        std::for_each(solve_threads.begin(),solve_threads.end(),[](thread& solve_thread){
          if(true == solve_thread.joinable())solve_thread.join();
        });
        solve_threads.clear();
      }

      /* Store the current data and move the iterator forward for the next one */
      neuron_data.step();
    }
  }else throw "A solution of 0 rows!";
}

void Solution_solver::solve_a_partial(const vector<sdouble32>& input, uint32 row_iterator, uint32 col_iterator){
  partial_solvers[row_iterator][col_iterator].collect_input_data(input);
  partial_solvers[row_iterator][col_iterator].solve();
  partial_solvers[row_iterator][col_iterator].provide_output_data();
  partial_solvers[row_iterator][col_iterator].provide_gradient_data(
    transfer_function_input, transfer_function_output
  );
}

} /* namespace sparse_net_library */

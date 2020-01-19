#include "services/solution_solver.h"
#include "services/synapse_iterator.h"

#include <thread>

namespace sparse_net_library{

using std::swap_ranges;

Solution_solver::Solution_solver(
  const Solution& to_solve, Service_context context
): solution(to_solve){
  number_of_threads = context.get_max_solve_threads();
  partial_solvers = vector<vector<Partial_solution_solver>>(solution.cols_size());
  partial_solver_output_maps = vector<vector<Synapse_iterator>>(solution.cols_size());
  neuron_data = vector<sdouble32>(solution.neuron_number());
  for(int row_iterator = 0; row_iterator < solution.cols_size(); ++row_iterator){
    partial_solvers[row_iterator] = vector<Partial_solution_solver>(
      solution.cols(row_iterator), Partial_solution_solver(get_partial(0,0,solution))
    );
    for(uint32 column_index = 0; column_index < solution.cols(row_iterator); ++column_index){
      partial_solvers[row_iterator][column_index] = Partial_solution_solver(
        get_partial(row_iterator,column_index,solution)
      ); /* Initialize a solver for this partial solution element */
      partial_solver_output_maps[row_iterator].push_back(Synapse_iterator(
        get_partial(row_iterator,column_index,solution).output_data()
      )); /* Initialize a solver and output map for this partial @Partial_solution element */
    }
  } /* loop through every partial solution and initialize solvers and output maps for them */
}

vector<sdouble32> Solution_solver::solve(vector<sdouble32> input){

  using std::thread;
  using std::ref;
  using std::min;

  if(0 < solution.cols_size()){
    uint32 partial_solution_start;
    vector<thread> solve_threads;
    uint32 col_iterator;

    for(int row_iterator = 0; row_iterator < solution.cols_size(); ++row_iterator){
      partial_solution_start = 0;
      col_iterator = 0;
      if(0 == solution.cols(row_iterator)) throw "A solution row of 0 columns!";
      while(col_iterator < solution.cols(row_iterator)){
        for(uint16 i = 0; i < number_of_threads; ++i){
          if(col_iterator < solution.cols(row_iterator)){
            solve_threads.push_back(thread(&Solution_solver::solve_a_partial,
              this, ref(input), row_iterator, col_iterator, partial_solution_start
            ));
            partial_solution_start += get_partial(row_iterator,col_iterator,solution).internal_neuron_number();
            ++col_iterator;
          }else break;
        }
        std::for_each(solve_threads.begin(),solve_threads.end(),[](thread& solve_thread){
          if(true == solve_thread.joinable())solve_thread.join();
        });
        solve_threads.clear();
      }
    }
    return {neuron_data.end() - solution.output_neuron_number(),neuron_data.end()}; /* Return with the data of the last row */
  }else throw "A solution of 0 rows!";
}

void Solution_solver::solve_a_partial(vector<sdouble32>& input, uint32 row_iterator, uint32 col_iterator, uint32 partial_solution_start){

  using std::min;
  using std::max;

  vector<sdouble32> collected_output;
  uint32 output_iterator = 0;
  partial_solvers[row_iterator][col_iterator].collect_input_data(input,neuron_data); /* Collect the input for the partial solution solver */
  collected_output = partial_solvers[row_iterator][col_iterator].solve(); /* Run the partial solution solver */

  partial_solver_output_maps[row_iterator][col_iterator].skim([&](int partial_output_synapse_starts, unsigned int partial_output_synapse_size){
    swap_ranges( /* Save output into the internal neuron memory */
      collected_output.begin() + output_iterator,
      collected_output.begin() + output_iterator + partial_output_synapse_size,
      neuron_data.begin() + partial_output_synapse_starts
    );
    output_iterator += partial_output_synapse_size;
  });
}

} /* namespace sparse_net_library */

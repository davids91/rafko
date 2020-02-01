#include "services/solution_solver.h"
#include "services/synapse_iterator.h"

#include <thread>

namespace sparse_net_library{

using std::swap_ranges;

Solution_solver::Solution_solver(const Solution& to_solve, Service_context context)
: solution(to_solve)
, partial_solvers(solution.cols_size())
, partial_solver_output_maps(solution.cols_size())
, neuron_data(solution.neuron_number())
, transfer_function_input(solution.output_neuron_number())
, transfer_function_output(solution.output_neuron_number())
, number_of_threads(context.get_max_solve_threads())
{
  int partial_solution_end_plus1 = 0;
  int first_output_neuron_index = solution.neuron_number() - solution.output_neuron_number();
  for(int row_iterator = 0; row_iterator < solution.cols_size(); ++row_iterator){
    partial_solvers[row_iterator] = vector<Partial_solution_solver>(
      solution.cols(row_iterator), Partial_solution_solver(get_partial(0,0,solution))
    );
    for(uint32 column_index = 0; column_index < solution.cols(row_iterator); ++column_index){
      partial_solution_end_plus1 += get_partial(row_iterator,column_index,solution).internal_neuron_number();

      partial_solvers[row_iterator][column_index] = Partial_solution_solver(
        get_partial(row_iterator,column_index,solution),
        std::min( /* Transfer function data to be monitored for the output Neurons only */
          std::max(0,(partial_solution_end_plus1 - first_output_neuron_index + 1)),
          static_cast<int>(get_partial(row_iterator,column_index,solution).internal_neuron_number())
        ),
        context /* The session configuration */
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
    uint32 gradient_data_start = 0;
    vector<thread> solve_threads;
    uint32 col_iterator;

    for(int row_iterator = 0; row_iterator < solution.cols_size(); ++row_iterator){
      col_iterator = 0;
      if(0 == solution.cols(row_iterator)) throw "A solution row of 0 columns!";
      while(col_iterator < solution.cols(row_iterator)){
        for(uint16 i = 0; i < number_of_threads; ++i){
          if(col_iterator < solution.cols(row_iterator)){
            solve_threads.push_back(thread(&Solution_solver::solve_a_partial,
              this, ref(input), row_iterator, col_iterator, gradient_data_start
            ));
            gradient_data_start += partial_solvers[row_iterator][col_iterator].get_gradient_data_size();
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


void Solution_solver::solve_a_partial(vector<sdouble32>& input, uint32 row_iterator, uint32 col_iterator, uint32 gradient_data_start){

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

  if(0 < partial_solvers[row_iterator][col_iterator].get_gradient_data_size()){
    swap_ranges( /* In case it's applicable save data required for gradient calculation as well */
      transfer_function_input.begin() + gradient_data_start,
      transfer_function_input.begin() + gradient_data_start + partial_solvers[row_iterator][col_iterator].get_gradient_data_size(),
      partial_solvers[row_iterator][col_iterator].get_transfer_function_input().begin()
    );
    swap_ranges(
      transfer_function_output.begin() + gradient_data_start,
      transfer_function_output.begin() + gradient_data_start + partial_solvers[row_iterator][col_iterator].get_gradient_data_size(),
      partial_solvers[row_iterator][col_iterator].get_transfer_function_output().begin()
    );
  }
}

} /* namespace sparse_net_library */

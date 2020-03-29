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

#include "sparse_net_global.h"

#include <vector>
#include <atomic>

#include "gen/solution.pb.h"
#include "models/service_context.h"
#include "services/partial_solution_solver.h"

namespace sparse_net_library{

using std::vector;

/**
 * @brief      This class Processes a @Solution given in its constructor and handles
 *             the distribution of the needed resources for it.
 */
class Solution_solver{
public:
  Solution_solver(const Solution& to_solve, Service_context context = Service_context());

  /**
   * @brief      Solves the Solution given in the constructor, considering the previous runs
   *
   * @param[in]  input  The input data to be taken
   */
  void solve(vector<sdouble32> input);

  /**
   * @brief      Gets the output size of the solution. Th solution output is defined as the last
   *             Neurons in the solution. Cardinality is given by this function.
   *
   * @return     Number of output Neurons
   */
  sdouble32 get_output_size(void){
    return solution.output_neuron_number();
  }

  /**
   * @brief      Gets the raw input added into the transfer function, provided the @Partial_solution monitors for it
   *
   * @return     The array for the input values for the neurons.
   */
  const vector<sdouble32>& get_transfer_function_input(void) const{
     return transfer_function_input;
  }

  /**
   * @brief      Gets the output from to the transfer function, provided the @Partial_solution monitors for it
   *
   * @return     The array for the input values for the neurons.
   */
  const vector<sdouble32>& get_transfer_function_output(void) const{
    return transfer_function_output;
  }

  /**
   * @brief      Gets the neuron data.
   *
   * @return     The neuron data.
   */
  const vector<sdouble32>& get_neuron_data(void) const{
    return neuron_data[0];
  }

  /**
   * @brief      Resets Neuron data in the solver and in the partial solutions
   */
  void reset(void){
    for(sdouble32& neuron_data_element : neuron_data[0])neuron_data_element = 0;
    for(vector<Partial_solution_solver>& solver_row : partial_solvers)
      for(Partial_solution_solver& solver : solver_row)solver.reset();
  }

  /**
   * @brief      Gets the raw input added into the transfer function, provided the @Partial_solution monitors for it
   *
   * @param[in]  neuron_index  The neuron index
   *
   * @return     The array for the input values for the neurons.
   */
  sdouble32 get_transfer_function_input(uint32 neuron_index) const{
    if(solution.neuron_number() > neuron_index)return transfer_function_input[neuron_index];
      else throw "Neuron index out of bounds!";
  }

  /**
   * @brief      Gets the output from to the transfer function, provided the @Partial_solution monitors for it
   *
   * @param[in]  neuron_index  The neuron index
   *
   * @return     The array for the input values for the neurons.
   */
  sdouble32 get_transfer_function_output(uint32 neuron_index) const{
    if(solution.neuron_number() > neuron_index)return transfer_function_output[neuron_index];
      else throw "Neuron index out of bounds!";
  }

  /**
   * @brief      Gets the neuron data at the given neuron index
   *
   * @param[in]  index  Neuron index
   *
   * @return     The neuron data.
   */
  sdouble32 get_neuron_data(uint32 index) const{
    if(neuron_data[0].size() > index)return neuron_data[0][index];
     else throw "Neuron index out of bounds!";
  }

private:

  /**
   * @brief      Gets a @Partial_Solution reference from the solution based on the given coordinates.
   *
   * @param[in]  row       The row
   * @param[in]  col       The col
   * @param[in]  solution  The solution
   *
   * @return     The @Partial_solution reference.
   */
  static const Partial_solution& get_partial(uint32 row, uint32 col, const Solution& solution){
    if(solution.cols_size() <= static_cast<int>(row)) throw "Row index out of bounds!";
    uint32 index = 0;
    for(uint32 i = 0;i < row; ++i) index += solution.cols(i);
    return solution.partial_solutions(index + col);
  }

  /**
   * @brief      Utility to help solve a @Partial_solution
   *
   * @param      input                   The input
   * @param[in]  row_iterator            The row iterator
   * @param[in]  col_iterator            The col iterator
   */
  void solve_a_partial(vector<sdouble32>& input, uint32 row_iterator, uint32 col_iterator);

  const Solution& solution;
  vector<vector<Partial_solution_solver>> partial_solvers;
  vector<vector<sdouble32>> neuron_data;  /* The internal Data of each Neuron */
  vector<sdouble32> transfer_function_input; /* Extended data required for output layer error */
  vector<sdouble32> transfer_function_output;

  uint16 number_of_threads = 1;
};

} /* namespace sparse_net_library */

#endif /* SOLUTION_SOLVER_H */

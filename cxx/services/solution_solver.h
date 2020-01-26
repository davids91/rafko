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
   *
   * @return     The resulting output of the SparseNet.
   */
  vector<sdouble32> solve(vector<sdouble32> input);

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
    return neuron_data;
  }

  /**
   * @brief      Resets Neuron data in the solver and in the partial solutions
   */
  void reset(void){
    for(sdouble32& neuron_data_element : neuron_data)neuron_data_element = 0;
    for(vector<Partial_solution_solver>& solver_row : partial_solvers)
      for(Partial_solution_solver& solver : solver_row)solver.reset();
  }

  /**
   * @brief      Gets the neuron data at the given neuron index
   *
   * @param[in]  index  Neuron index
   *
   * @return     The neuron data.
   */
  sdouble32 get_neuron_data(uint32 index) const{
    if(neuron_data.size() > index)return neuron_data[index];
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

  void solve_a_partial(vector<sdouble32>& input, uint32 row_iterator, uint32 col_iterator, uint32 partial_solution_start);

  const Solution& solution;
  vector<vector<Partial_solution_solver>> partial_solvers;
  vector<vector<Synapse_iterator>> partial_solver_output_maps;  /* Maps each output of the partial solvers into an index in @neuron_data */
  vector<sdouble32> neuron_data;  /* The internal Data of each Neuron */
  uint16 number_of_threads = 1;
};

} /* namespace sparse_net_library */

#endif /* SOLUTION_SOLVER_H */

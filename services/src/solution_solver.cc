#include "services/solution_solver.h"

#include "services/synapse_iterator.h"

namespace sparse_net_library{

using std::vector;

vector<sdouble32> Solution_solver::solve(const Solution& solution, vector<sdouble32> input){
  if(0 < solution.output_number()){
    vector<sdouble32> result(solution.output_number());
    for(uint32 row_iterator = 0;row_iterator < solution.rows(); ++row_iterator){
      /* Collect the data for a row of partial solutions */
      Synapse_iterator iter();
      /* Run the partial solutions */
    }
    return result;
  }else throw "Solution of 0 outputs!";
}
} /* namespace sparse_net_library */

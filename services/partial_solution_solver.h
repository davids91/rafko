#ifndef Partial_solution_H
#define Partial_solution_H

#include <vector>

#include "sparse_net_global.h"
#include "models/gen/sparse_net.pb.h"
#include "models/gen/solution.pb.h"

namespace sparse_net_library {

using std::vector;

class Partial_solution_solver{

public:
  /**
   * @brief      Solves the detail given in the argument, then cleans it up and returns the solution
   *
   * @return     The result datas of the internal neurons
   */
  vector<sdouble32> solve(const Partial_solution* detail, const vector<sdouble32>* input_data);

  /**
   * @brief      Determines if given Solution Detail is valid. Due to performance reasons
   *             this function isn't used while solving a SparseNet
   *
   * @return     True if detail is valid, False otherwise.
   */
  bool is_valid(const Partial_solution* detail);

private:
  vector<sdouble32> data; /* size of @internal_neuron_number + @input_data_size  */

};

} /* namespace sparse_net_library */
#endif /* Partial_solution_H */

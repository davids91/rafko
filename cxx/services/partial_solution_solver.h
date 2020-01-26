#ifndef Partial_solution_H
#define Partial_solution_H

#include <vector>

#include "sparse_net_global.h"

#include "gen/sparse_net.pb.h"
#include "gen/solution.pb.h"
#include "services/synapse_iterator.h"

namespace sparse_net_library {

using std::vector;
using std::reference_wrapper;

class Partial_solution_solver{

public:
  Partial_solution_solver(const Partial_solution& partial_solution)
  : detail(partial_solution)
  , internal_iterator(detail.get().inside_indices())
  , input_iterator(partial_solution.input_data())
  { reset(); }

  /**
   * @brief      Gets the size of the elements taken by the configurad Patial solution.
   *
   * @return     The input size in number of elements ( @sdouble32 ).
   */
  uint32 get_input_size(void) const;
  void collect_input_data(vector<sdouble32>& input_data, vector<sdouble32> neuron_data);

  /**
   * @brief      Solves the detail given in the argument, then cleans it up and returns the solution
   *
   * @return     The result data of the internal neurons
   */
  vector<sdouble32> solve();

  /**
   * @brief      Resets the data of the included Neurons.
   */
  void reset(void);

  /**
   * @brief      Determines if given Solution Detail is valid. Due to performance reasons
   *             this function isn't used while solving a SparseNet
   *
   * @return     True if detail is vmakealid, False otherwise.
   */
  bool is_valid(void);

private:
  reference_wrapper<const Partial_solution> detail;
  Synapse_iterator internal_iterator;
  Synapse_iterator input_iterator;
  vector<sdouble32> neuron_output;
  vector<sdouble32> collected_input_data;

};

} /* namespace sparse_net_library */
#endif /* Partial_solution_H */

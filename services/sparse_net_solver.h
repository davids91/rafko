#ifndef sparse_netsolver_H
#define sparse_netsolver_H

#include <memory>

#include "sparse_net_global.h"
#include "models/gen/sparse_net.pb.h"

namespace sparse_net_library {

using std::vector;
using std::unique_ptr;

class SparseNetSolver
{
public:
  static unique_ptr<sdouble32> solve(const SparseNet *net);
  static unique_ptr<sdouble32> solve(vector<sdouble32> output, const SparseNet* net);

  static unique_ptr<sdouble32> calculate_spikes(const SparseNet *net);
  static unique_ptr<sdouble32> calculate_spikes(vector<sdouble32> output, const SparseNet* net);
private:
  /**
   * @brief The solution_state struct stores the states of a neuron
   * required for solving a neural network
   */
  struct Solution_state{
    /**
     * @brief solution_range_*: encapsulates the minimal enclosed range the neuron inputs are coherent
     */
    uint32 solution_range_start;
    uint32 solution_range_end;

    /**
     * @brief finished_operations stores the status of every neuron in range of the current solution
     */
    vector<uint32> neuron_finished_operations; /* number of inputs + itself ; number of inputs + 2 means the neuron is locked for a detail operation */
  };
};

} /* namespace sparse_net_library */
#endif /* sparse_netsolver_H */

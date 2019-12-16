#ifndef snetsolver_H
#define snetsolver_H

#include <memory>

#include "sparsenet_global.h"
#include "models/sNet.pb.h"

namespace sparse_net_library {

class snetsolver
{
public:
  static std::unique_ptr<sdouble32> solve(const SparseNet *net);
  static std::unique_ptr<sdouble32> solve(std::unique_ptr<sdouble32[]> output, const SparseNet *net);

  static std::unique_ptr<sdouble32> calculateSpikes(const SparseNet *net);
  static std::unique_ptr<sdouble32> calculateSpikes(std::unique_ptr<sdouble32[]> output, const SparseNet *net);
private:

  /**
   * @brief The solution_state struct stores the states of a neuron
   * required for solving a neural network
   */
  struct solution_state{
    /**
     * @brief solution_range_*: encapsulates the minimal enclosed range the neuron inputs are coherent
     */
    uint32 solution_range_start;
    uint32 solution_range_end;

    /**
     * @brief finished_operations stores the status of every neuron in range of the current solution
     */
    std::vector<uint32> neuron_finished_operations; /* number of inputs + itself ; number of inputs + 2 means the neuron is locked for a detail operation */
  };

  /**
   * @brief The detail_internal_neuron struct is a representation of the actual neuron to be used in an intermediate solution
   */
  struct detail_internal_neuron{ /** TODO: no structures, only arrays! */
    uint32 actual_index;
    uint32 input_size;
    std::unique_ptr<uint8[]> inside_indexes; /* Ranges [0;input_data_size + detail_internal_neuron_num), the input the neuron shal take */
    std::unique_ptr<sdouble32[]> weights; /* Weights for @inside_indexes */
    transfer_functions transfer_function;
    sdouble32 memory_ratio;
    sdouble32 bias;
  };

  /**
   * @brief The solution_detail struct represents an intermediate solution
   * as it calculates a coherent part of the net, where locality is maximized
   */
  struct solution_detail{
    /** Solution data */
    uint8 detail_internal_neuron_num; /* number of neurons used in this part */
    uint32 input_data_size; /* number of used inputs */
    std::unique_ptr<sdouble32[]> data; /* size of @detail_internal_neuron_num + @input_data_size  */
    std::unique_ptr<bool[]> neuron_state; /* size of @detail_internal_neuron_num */
    std::unique_ptr<detail_internal_neuron[]> neuron_table; /* size of @detail_internal_neuron_num */

  };
};

} /* namespace sparse_net_library */
#endif // snetsolver_H

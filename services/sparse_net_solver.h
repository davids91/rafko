#ifndef snetsolver_H
#define snetsolver_H

#include <memory>

#include "sparsenet_global.h"
#include "models/sNet.pb.h"
#include "test/test_mockups.h"

namespace sparse_net_library {

class SparseNetSolver
{
public:
  static std::unique_ptr<sdouble32> solve(const SparseNet *net);
  static std::unique_ptr<sdouble32> solve(std::vector<sdouble32> output, const SparseNet *net);

  static std::unique_ptr<sdouble32> calculate_spikes(const SparseNet *net);
  static std::unique_ptr<sdouble32> calculate_spikes(std::vector<sdouble32> output, const SparseNet *net);
private:
  friend class sparse_net_library_test::Solver_mockup;
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
    std::vector<uint32> neuron_finished_operations; /* number of inputs + itself ; number of inputs + 2 means the neuron is locked for a detail operation */
  };

  /**
   * @brief The solution_detail struct represents an intermediate solution
   * as it calculates a coherent part of the net, where locality is maximized.
   * The arrays inside this structure all have their sizes defined by @internal_neuron_number
   * and/or @input_data_size.
   */
  struct Solution_detail{
    /**
     * #################################################################################################
     * Solution data
     * #################################################################################################
     */
    uint16 internal_neuron_number; /* number of neurons used in this part */
    uint32 input_data_size; /* number of used inputs */
    std::vector<sdouble32> data; /* size of @internal_neuron_number + @input_data_size  */

    /**
     * #################################################################################################
     * Detail Internal Neuron:
     * a representation of the actual neuron to be used in an intermediate solution
     * - size is of @internal_neuron_number
     * #################################################################################################
     */
    /**
     * stores the actual index of the Neuron it calculates the value for
     * - size is of @internal_neuron_number
     * - Indexes are global to @neuron_array under the @SparseNet to be solved
     */
    std::vector<uint32> actual_index;

    /**
     * stores how many inputs each neuron has
     * - size is of @internal_neuron_number
     */
    std::vector<uint32> input_sizes;

    /**
     * Parameters each neuron have to post-process their inputs
     * - sizes are of @internal_neuron_number
     */
    std::vector<transfer_functions> transfer_functions;
    std::vector<sdouble32> memory_ratios;
    std::vector<sdouble32> biases;

    /**
     * #################################################################################################
     * Indexes and Weights:
     * Stores the inputs and their corresponding weights of the Neurons
     * - size are of the summary of @input_sizes
     * - Since solving the detail is incremental, the start of each Neuron's
     *   input is determined at runtime, so it doesn't need to be stored
     * #################################################################################################
     */
    /**
     * stores the number of induvidual inputs of the Neurons
     * - Ranges [0;@input_data_size + @internal_neuron_number)
     * - Indexes are local to solution detail
     */
    std::vector<uint16> inside_indexes;

    /**
     * stores the weights paired to @inside_indexes for the inputs of the Neurons
     * - Ranges [0.0,1.0)
     */
    std::vector<sdouble32> weights;
  };

  /**
   * @brief      Solves the detail given in the argument, then cleans it up and returns the solution
   *
   * @param[in]  detail  Already initialized solution detail
   * @param[in]  info    The allocated space of the neuron data, or nullpointer ( signals to allocate the data )
   *
   * @return     The result datas of the internal neurons
   */
  std::vector<sdouble32> solveDetail(std::unique_ptr<Solution_detail> detail);

  /**
   * @brief      Determines if given Solution Detail is valid. Due to performance reasons
   *             this function isn't used while solving the SparseNet
   *
   * @param      detail  The detail
   *
   * @return     True if detail is valid, False otherwise.
   */
  bool is_detail_valid(Solution_detail* detail);
};

} /* namespace sparse_net_library */
#endif /* snetsolver_H */

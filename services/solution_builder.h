#ifndef SOLUTION_CHAIN_BUILDER_H

#include <memory>
#include <deque>
#include <vector>
#include <atomic>

#include "models/gen/sparse_net.pb.h"
#include "models/gen/solution.pb.h"

#include "sparse_net_global.h"

namespace sparse_net_library {

using std::unique_ptr;
using std::vector;
using std::deque;
using std::atomic;

/**
 * @brief      Front-end to create a @Soltuion to solve a @SparseNet. @max_solve_threads
 *             determines the maximum number of threads to be used inside the @build function.
 *             A @Solution_chain is built up by an array of @Decoupled_solutions. The first
 *             @Decoupled_solutions are independent because they are mostly processing only
 *             inputs to the Neural network. Any further @Partial_solution messages
 *             depend on the @Decoupled_solutions under the previous index.
 *             In case there is only one used device for the net, which has insufficient internal
 *             memory for a whole @Decoupled_solutions item, the items in there can be executed
 *             sequentally.
 *             The separation of the net into decoupled independent partial solutions enable
 *             distributed computing based on micro-services, as the elements inside @Decoupled_solutions
 *             can be solved in an independent manner. Dependencies inside the Neural network are represented
 *             in the order of the elements in a @solution_chain.
 */
class Solution_builder{
public:
  /**
   * @brief      Set the number of threads to be used while generating the solution to the given @SparseNet
   *
   * @param[in]  number  The number of threads to be used
   *
   * @return     Builder reference for chaining
   */
  Solution_builder& max_solve_threads(uint8 number);
  Solution_builder& device_max_megabytes(sdouble32 megabytes);

  /**
   * @brief      Set the used arena pointer
   *
   * @param      arena  The arena
   *
   * @return     Builder reference for chaining
   */
  Solution_builder& arena_ptr(google::protobuf::Arena* arena);

  /**
   * @brief      Build the Solution to be solved by @Solution_solver
   *
   * @param[in]  net   The net
   *
   * @return     Builder reference for chaining
   */
  Solution* build( SparseNet& net );

private:
  /**
   * Helper variables to see if different required arguments are set inside the builder
   */
  bool is_max_solve_threads_set = false;
  google::protobuf::Arena* arg_arena_ptr = nullptr;
  uint8 arg_max_solve_threads = 1;
  sdouble32 arg_device_max_megabytes = 2.0 /* GB */ * 1024.0/* MB */;

  /**
   * Number of already processed output layer Neurons
   */
  atomic<uint32> output_layer_iterator;

  /**
   * For each @Neuron in @SparseNet stores the processed state. Values:
   *  - Number of processed children ( storing raw children number without partition information )
   *  - Number of processed children + 1 in case the Neuron is reserved
   *  - Number of processed children + 2 in case the Neuron is processed
   */
  vector<unique_ptr<atomic<uint32>>> neuron_states;

  /**
   * Number of inputs a Neuron has, based on the input index partition sizes
   */
  vector<uint32> neuron_number_of_inputs;

  /**
   * A subset of the net representing independent solutions
   */
  std::mutex net_subset_mutex;
  std::atomic<sdouble32> net_subset_size; /* The size of the currently partial solution to be built in bytes */
  deque<uint32> net_subset_index;
  deque<uint32> net_subset; /*!#4 add transitively dependent neurons if memory allows it */

  /**
   * @brief      Builds a thread.
   *
   * @param      net           The net
   * @param      result        The result
   * @param[in]  thread_index  The thread index
   */
  void collect_subset_thread( SparseNet& net, Solution& result, uint8 thread_index);


  /**
   * @brief      Gets the partial solution from subset.
   *
   * @param      net                      The net
   * @param      solution_neuron_indexes  The solution neuron indexes
   *
   * @return     The partial solution from subset.
   */
  unique_ptr<Decoupled_solutions> get_partial_solution_from_subset( SparseNet& net, deque<uint32>& solution_neuron_indexes);

  /**
   * @brief      Inline functions to help build partial solutions
   *
   * @param[in]  neuron_index  The neuron index inside @neuron_number_of_inputs and @neuron_states
   *
   * @return     Information depending on the function
   */
  inline bool is_neuron_in_progress(uint32 neuron_index) const{
    return (*neuron_states[neuron_index] < neuron_number_of_inputs[neuron_index] + 1);
  }
  inline bool is_neuron_reserved(uint32 neuron_index) const{
    return (*neuron_states[neuron_index] < neuron_number_of_inputs[neuron_index] + 1);
  }
  inline bool is_neuron_solvable(uint32 neuron_index) const{
    return (neuron_number_of_inputs[neuron_index] == *neuron_states[neuron_index]);
  }
  inline bool is_neuron_processed(uint32 neuron_index) const{
    return (*neuron_states[neuron_index] < neuron_number_of_inputs[neuron_index] + 2);
  }
};

} /* namespace sparse_net_library */

#endif /* SOLUTION_CHAIN_BUILDER_H */

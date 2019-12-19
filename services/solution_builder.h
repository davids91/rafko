#ifndef SOLUTION_CHAIN_BUILDER_H

#include <memory>
#include <deque>
#include <vector>
#include <atomic>

#include "models/gen/sparse_net.pb.h"
#include "models/gen/solution.pb.h"
#include "services/neuron_router.h"

#include "sparse_net_global.h"

namespace sparse_net_library {

/**
 * @brief      Front-end to create a @Soltuion to solve a @SparseNet. @max_solve_threads
 *             determines the maximum number of threads to be used inside the @build function.
 *             A @Solution_chain is built up by a 2D array of @Partial_solutions. The first row
 *             is independent because they are mostly processing only inputs to the Neural network
 *             and Neurons driectly dependent on them.
 *             Any further @Partial_solution messages depend on the @Partial_solution in thep previous row.
 *             In case there is only one used device for the net, which has insufficient internal
 *             memory for a big @Partial_solution, it can be divided into multiple smaller ones,
 *             which are executed sequentially.
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
  Solution* build(SparseNet& net);

private:
  /**
   * Helper variables to see if different required arguments are set inside the builder
   */
  bool is_max_solve_threads_set = false;
  google::protobuf::Arena* arg_arena_ptr = nullptr;
  uint8 arg_max_solve_threads = 1;
  sdouble32 arg_device_max_megabytes = 2.0 /* GB */ * 1024.0/* MB */;

  /**
   * @brief      Adds a neuron to partial solution.
   *
   * @param      net  The sparse net to read the neuron from
   * @param      neuron_index  the index of the neuron inside the net
   * @param      partial  the partial solution reference to add the Neuron into
   *
   * @return     Return true in case the Neuron could b added to the @Partial_Solution
   */
  bool add_neuron_to_partial_solution(const SparseNet& net, uint32 neuron_index, Partial_solution& partial);

  /**
   * @brief      Checks for duplicates in thegiven @Partial_Solution, eliminates duplicates,
   *             and then corrects the indexes.
   *
   * @param      partial The @Partial_solution to correct
   *
   * @return     Returns with the memory usage reduction in Megabytes
   */
  uint32 check_for_duplicates_in_partial_solution(Partial_solution& partial);

  /**
   * @brief      Generates or adds to a @Partial_solution according to the current net Subset;
   *             Updates the states of the corresponding Neurons
   *
   * @param      net                      The Sparse net
   * @param      solution_neuron_indexes  The solution neuron indexes
   *
   * @return     The partial solution from subset.
   */
  void generate_partial_solution_from_subset(const SparseNet& net, Neuron_router& net_iterator, Partial_solution& current_partial);
};

} /* namespace sparse_net_library */

#endif /* SOLUTION_CHAIN_BUILDER_H */

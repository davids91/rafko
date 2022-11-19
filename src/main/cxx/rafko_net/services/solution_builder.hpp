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

#ifndef SOLUTION_BUILDER_H
#define SOLUTION_BUILDER_H

#include "rafko_global.hpp"

#include <memory>
#if(RAFKO_USES_OPENCL)
#include <string>
#endif/*(RAFKO_USES_OPENCL)*/

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/solution.pb.h"
#include "rafko_mainframe/models/rafko_settings.hpp"

namespace rafko_net {

/**
 * @brief      Front-end to create a @Soltuion to solve a @RafkoNet. @max_solve_threads
 *             determines the maximum number of threads to be used inside the @build function.
 *             A @Solution_chain is built up by a 2D array of @PartialSolutions. The first row
 *             is independent because they are mostly processing only inputs to the Neural network
 *             and Neurons driectly dependent on them.
 *             Any further @PartialSolution messages depend on the @PartialSolution in thep previous row.
 *             In case there is only one used device for the net, which has insufficient internal
 *             memory for a big @PartialSolution, it can be divided into multiple smaller ones,
 *             which are executed sequentially.
 *             The separation of the net into decoupled independent partial solutions enable
 *             distributed computing based on micro-services, as the elements inside @Decoupled_solutions
 *             can be solved in an independent manner. Dependencies inside the Neural network are represented
 *             in the order of the elements in a @solution_chain.
 */
class RAFKO_EXPORT SolutionBuilder{
public:

  /**
   * @brief      Constructs a new instance.
   *
   * @param[in]  settings  The Service settings
   */
  constexpr SolutionBuilder(const rafko_mainframe::RafkoSettings& settings)
  :  m_settings(settings)
  { }

  /**
   * @brief      Build the Solution to be solved by @SolutionSolver
   *
   * @param[in]  net               The network to build a solution from
   * @param[in]  arena_ptr         The pointer to the arena owning the generated Solution or nullptr if it should be on the heap
   * @param[in]  optimize_to_gpu   True, Should the resulting solution be optimized for large amount of threads
   *
   * @return     Builder reference for chaining
   */
  Solution* build(const RafkoNet& net, google::protobuf::Arena* arena_ptr, bool optimize_to_gpu = false);

  /**
   * @brief      Build the Solution to be solved by @SolutionSolver
   *
   * @param[in]  network           The network to build the generated solution from
   * @param[in]  optimize_to_gpu   True, Should the resulting solution be optimized for large amount of threads
   *
   * @return     Builder reference for chaining
   */
  Solution* build(const RafkoNet& network, bool optimize_to_gpu = false){
    return build(network, m_settings.get_arena_ptr(), optimize_to_gpu);
  }

  /**
   * @brief     Builds a Soltuion from the given netwrok reference and swaps it with another one
   *            This method aims to make it possible to generate multiple solutions without filling up
   *            an Arena endlessly ( Should there be any stored in the given settings instance ) by
   *            swapping the newly generated with the previous one.
   *
   * @param         previous            A pointer to the @Solution object to swap the newly generated solution with.
   *                                    The provided pointer is advised to be pointing to an object allocated on a protobuf Arena
   * @param[in]     network             The network to build the generated solution from
   * @param[in]     optimize_to_gpu     True, Should the resulting solution be optimized for large amount of threads
   */
  void update(Solution* previous, const RafkoNet& network, bool optimize_to_gpu = false);

  #if(RAFKO_USES_OPENCL)
  /**
   * @brief     Generate the OpenCL Kernel code solving the provided solution instance with the given parameters.
   *            The resulting kernel should be able to solve the solution for multiple inputs at the same time.
   *
   * @param[in]     solution              The Solution to base Kernel code generation upon
   * @param[in]     name                  The name of the resulting kernel
   * @param[in]     sequence_size         The number of consecutive input-label pairs an envirnment should have for one item
   * @param[in]     prefill_input_num     The number of prefill inputs the data set has
   * @param[in]     settings              The Setting instance containing some of the required parameters
   *
   * @return    The generated Kernel code to compile and send to an OpenCL Device
   */
  static std::string get_kernel_for_solution(
    const Solution& solution, std::string name, std::uint32_t sequence_size, std::uint32_t prefill_input_num,
    const rafko_mainframe::RafkoSettings& settings
  );
  #endif/*(RAFKO_USES_OPENCL)*/

private:
  const rafko_mainframe::RafkoSettings& m_settings;

  static std::uint32_t get_last_neuron_index_of_partial(const PartialSolution& partial){
    return (partial.output_data().starts() + partial.output_data().interval_size() - 1u);
  }

};

} /* namespace rafko_net */

#endif /* SOLUTION_BUILDER_H */

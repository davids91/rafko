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

#include "rafko_global.h"

#include <memory>
#include <vector>
#include <thread>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/solution.pb.h"
#include "rafko_mainframe/models/service_context.h"

namespace rafko_net {

using std::vector;
using std::thread;

using rafko_mainframe::ServiceContext;

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
class SolutionBuilder{
public:

  /**
   * @brief      Constructs a new instance.
   *
   * @param[in]  context  The Service context
   */
  SolutionBuilder(ServiceContext& service_context_)
  :  service_context(service_context_)
  { }

  /**
   * @brief      Build the Solution to be solved by @SolutionSolver
   *
   * @param[in]  net               The network to build a solution from
   * @param[in]  optimize_to_gpu   Should the resulting solution be optimized for large amount of threads
   *
   * @return     Builder reference for chaining
   */
  Solution* build(const RafkoNet& net, bool optimize_to_gpu = false);

private:
  ServiceContext& service_context;

  static sdouble32 get_size_in_mb(const PartialSolution& partial){
    return partial.SpaceUsedLong() /* Bytes */ / double_literal(1024.0) /* KB */ / double_literal(1024.0) /* MB */;
  }

  static uint32 get_last_neuron_index_of_partial(const PartialSolution& partial){
    return (partial.output_data().starts() + partial.output_data().interval_size() - 1u);
  }
};

} /* namespace rafko_net */

#endif /* SOLUTION_BUILDER_H */

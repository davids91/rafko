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

#include "sparse_net_global.h"

#include <memory>
#include <vector>
#include <deque>
#include <thread>

#include "gen/sparse_net.pb.h"
#include "gen/solution.pb.h"
#include "rafko_mainframe/models/service_context.h"

#include "sparse_net_library/services/neuron_router.h"
#include "sparse_net_library/services/synapse_iterator.h"

namespace sparse_net_library {

using std::vector;
using std::deque;
using std::thread;

using rafko_mainframe::Service_context;

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
   * @brief      Constructs a new instance.
   *
   * @param[in]  context  The Service context
   */
  Solution_builder(Service_context& context_)
  :  context(context_)
  { }

  /**
   * @brief      Build the Solution to be solved by @Solution_solver
   *
   * @param[in]  net   The net
   *
   * @return     Builder reference for chaining
   */
  Solution* build(const SparseNet& net);

private:
  Service_context& context;

  static sdouble32 get_size_in_mb(const Partial_solution& partial){
    return partial.SpaceUsedLong() /* Bytes */ / double_literal(1024.0) /* KB */ / double_literal(1024.0) /* MB */;
  }
};

} /* namespace sparse_net_library */

#endif /* SOLUTION_BUILDER_H */

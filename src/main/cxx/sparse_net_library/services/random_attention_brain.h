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

#ifndef RANDOM_ATTENTION_BRAIN_H
#define RANDOM_ATTENTION_BRAIN_H

#include "rafko_global.h"

#include <vector>

#include "gen/sparse_net.pb.h"
#include "gen/solution.pb.h"

#include "rafko_mainframe/models/service_context.h"
#include "sparse_net_library/models/data_aggregate.h"

#include "sparse_net_library/services/Solution_solver.h"
#include "sparse_net_library/services/weight_updater.h"
#include "sparse_net_library/services/weight_experience_space.h"

namespace sparse_net_library{

using std::vector;
using std::unique_ptr;

using rafko_mainframe::Service_context;

class Random_attention_brain{
public:
  Random_attention_brain(SparseNet& neural_network, Data_aggregate& training_set_, Service_context& service_context);

  /**
   * @brief      Add an impulse to a random weight based on the Networks performance on the given dataset
   */
  void step(void);

  /**
   * @brief      For the given weight, it returns with its experience space
   *
   * @param[in]  weight_index  The weight index
   *
   * @return     The weight experiences.
   */
  const Weight_experience_space& get_weight_experiences(uint32 weight_index) const;

private:
  SparseNet& net;
  Service_context& context;
  Solution* net_solution;
  vector<unique_ptr<Solution_solver>> solvers;
  Weight_updater weight_updater;
  Data_aggregate& training_set;
  uint32 memory_truncation;
  vector<Weight_experience_space> weightxp_space;
  uint32 iteration;
};

} /* namespace sparse_net_library */

#endif /*  RANDOM_ATTENTION_BRAIN_H */
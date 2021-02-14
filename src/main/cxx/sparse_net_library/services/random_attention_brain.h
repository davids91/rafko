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
#include <mutex>

#include "gen/sparse_net.pb.h"
#include "gen/solution.pb.h"

#include "rafko_mainframe/models/service_context.h"
#include "sparse_net_library/models/data_aggregate.h"

#include "sparse_net_library/services/Solution_solver.h"
#include "sparse_net_library/services/weight_updater.h"
#include "sparse_net_library/services/weight_experience_space.h"

namespace sparse_net_library{

using std::vector;
using std::mutex;
using std::unique_ptr;

using rafko_mainframe::Service_context;

class Random_attention_brain{
public:
  Random_attention_brain(SparseNet& neural_network, Data_aggregate& training_set_, Service_context& service_context);

  /**
   * @brief      Add an impulse to a random weight based on the Networks performance on the given dataset
   */
  void step(void);

private:
  SparseNet& net;
  Service_context& context;
  Solution* net_solution;
  vector<thread> solve_threads;
  vector<unique_ptr<Solution_solver>> solvers;
  Weight_updater weight_updater;
  Data_aggregate& training_set;
  uint32 memory_truncation;
  vector<Weight_experience_space> weightxp_space;
  mutex dataset_mutex;

  /**
   * @brief      Evaluate the configured network on the given data set, which updates the stored error states of the set
   *
   * @param[in]  label_start_index            The raw label start
   * @param[in]  labels_to_eval               The labels to eval
   * @param[in]  start_index_inside_sequence  The first index in the sequence to start evaluating from up until the end or however long memory trunation lets it
   * @param[in]  sequence_truncation          The number of labels to be evaluated inside sequences
   */
  void evaluate(
    uint32 label_start_index, uint32 labels_to_eval,
    uint32 start_index_inside_sequence, uint32 sequence_truncation
  );

  /**
   * @brief      A thread to calcualte the error value for the given data set
   *
   * @param[in]  solve_thread_index           The index of the used thread
   * @param[in]  sequence_start_index         The starting sequence index to evaluate in this
   * @param[in]  sequences_to_evaluate        The sequences to evaluate
   * @param[in]  start_index_inside_sequence  The first index in the sequence to start evaluating from up until the end or however long memory trunation lets it
   * @param[in]  sequence_truncation          The number of labels to be evaluated inside sequences
   */
  void evaluate_thread(
    uint32 solve_thread_index, uint32 sequence_start_index, uint32 sequences_to_evaluate,
    uint32 start_index_inside_sequence, uint32 sequence_truncation
  );
};

} /* namespace sparse_net_library */

#endif /*  RANDOM_ATTENTION_BRAIN_H */
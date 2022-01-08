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

#ifndef RAFKO_CPU_CONTEXT_H
#define RAFKO_CPU_CONTEXT_H

#include "rafko_global.h"

#include <memory>

#include "rafko_net/services/solution_solver.h"
#include "rafko_gym/models/rafko_environment.h"
#include "rafko_gym/models/rafko_objective.h"
#include "rafko_gym/models/rafko_agent.h"

#include "rafko_mainframe/services/rafko_context.h"

namespace rafko_mainframe {

class RAFKO_FULL_EXPORT RafkoCPUContext : public RafkoContext{
public:

  RafkoCPUContext(rafko_net::RafkoNet& neural_network, rafko_mainframe::RafkoSettings settings_);
  ~RafkoCPUContext() = default;

  void set_environment(std::unique_ptr<rafko_gym::RafkoEnvironment> environment_);
  const rafko_gym::RafkoEnvironment& get_environment(){
    return *environment;
  }
  void set_objective(std::unique_ptr<rafko_gym::RafkoObjective> objective_);
  const rafko_gym::RafkoObjective& get_objective(){
    return *objective;
  }
  void set_weight_updater(std::unique_ptr<rafko_gym::RafkoWeightUpdater> weight_updater_);

  rafko_gym::RafkoWeightUpdater& expose_weight_updater(){
    return *weight_updater;
  };

  sdouble32 full_evaluation(){
    return evaluate(
      0u, environment->get_number_of_sequences(),
      0u, environment->get_sequence_size()
    );
  }

  sdouble32 stochastic_evaluation(bool to_seed = false, uint32 seed_value = 0u){
    if(to_seed)srand(seed_value);
    uint32 sequence_start_index = (rand()%(environment->get_number_of_sequences() - used_minibatch_size + 1));
    uint32 start_index_inside_sequence = (rand()%( /* If the memory is truncated for the training.. */
      environment->get_sequence_size() - used_sequence_truncation + 1 /* ..not all result output values are evaluated.. */
    )); /* ..only settings.get_memory_truncation(), starting at a random index inside bounds */
    return evaluate(
      sequence_start_index, used_minibatch_size,
      start_index_inside_sequence, used_sequence_truncation
    );
  }

  void push_state(){
    environment->push_state();
  }

  void pop_state(){
    environment->pop_state();
  }

  rafko_mainframe::RafkoSettings& expose_settings(){
    return settings;
  }

  rafko_net::RafkoNet& expose_network(){
    return network;
  }

private:
  google::protobuf::Arena arena;
  rafko_mainframe::RafkoSettings settings;
  rafko_net::RafkoNet& network;
  std::unique_ptr<rafko_net::Solution> network_solution;
  std::unique_ptr<rafko_net::SolutionSolver> agent;
  std::unique_ptr<rafko_gym::RafkoEnvironment> environment;
  std::unique_ptr<rafko_gym::RafkoObjective> objective;
  std::unique_ptr<rafko_gym::RafkoWeightUpdater> weight_updater;

  std::vector<std::vector<sdouble32>> neuron_outputs_to_evaluate; /* for each feature array inside each sequence inside each thread in one evaluation iteration */
  rafko_utilities::ThreadGroup execution_threads;

  uint32 used_sequence_truncation;
  uint32 used_minibatch_size;

  /**
   * @brief      Evaluate the given data set with the given parameters
   *
   * @param[in]  sequence_start             The starting sequence to be evaluated inside the @data_set
   * @param[in]  sequences_to_evaluate      The number of sequences to evaluate inside the @data_set
   * @param[in]  start_index_in_sequence    Parameter for sequence truncation: only update error value starting from this index in every sequence
   * @param[in]  sequence_tructaion         The number of labels to evaluate inside every evaluated sequence
   * @return     The resulting fitness
   */
  sdouble32 evaluate(uint32 sequence_start, uint32 sequences_to_evaluate, uint32 start_index_in_sequence, uint32 sequence_tructaion);
};

} /* namespace rafko_mainframe */

#endif /* RAFKO_CPU_CONTEXT_H */

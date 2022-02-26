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
#include "rafko_gym/services/updater_factory.h"

#include "rafko_mainframe/services/rafko_context.h"

namespace rafko_mainframe {

class RAFKO_FULL_EXPORT RafkoCPUContext : public RafkoContext{
public:

  RafkoCPUContext(rafko_net::RafkoNet& neural_network_, rafko_mainframe::RafkoSettings settings_ = rafko_mainframe::RafkoSettings());
  ~RafkoCPUContext() = default;

  void fix_dirty(){ /*!Note: When another contex updates the weights this hack takes over the changes */
    weight_updater->update_solution_with_weights();
  }

  /* +++ Methods taken from @RafkoContext +++ */
  void set_environment(std::shared_ptr<rafko_gym::RafkoEnvironment> environment_);

  void set_objective(std::shared_ptr<rafko_gym::RafkoObjective> objective_){
    objective.reset();
    objective = objective_;
  }

  void set_weight_updater(rafko_gym::Weight_updaters updater){
    weight_updater.reset();
    weight_updater = rafko_gym::UpdaterFactory::build_weight_updater(network, *network_solution, updater, settings);
  }

  void set_network_weight(std::uint32_t weight_index, double weight_value){
    assert( static_cast<std::int32_t>(weight_index) < network.weight_table_size() );
    network.set_weight_table(weight_index, weight_value);
    weight_updater->update_solution_with_weights();
  };

  void set_network_weights(const std::vector<double>& weights){
    assert( static_cast<std::int32_t>(weights.size()) == network.weight_table_size() );
    *network.mutable_weight_table() = {weights.begin(), weights.end()};
    weight_updater->update_solution_with_weights();
  };

  void apply_weight_update(const std::vector<double>& weight_delta){
    assert( static_cast<std::int32_t>(weight_delta.size()) == network.weight_table_size() );
    if(weight_updater->is_finished())
      weight_updater->start();
    weight_updater->iterate(weight_delta);
    weight_updater->update_solution_with_weights();
  };

  double full_evaluation(){
    return evaluate(
      0u, environment->get_number_of_sequences(),
      0u, environment->get_sequence_size()
    );
  }

  double stochastic_evaluation(bool to_seed = false, std::uint32_t seed_value = 0u){
    if(to_seed)srand(seed_value);
    std::uint32_t sequence_start_index = (rand()%(environment->get_number_of_sequences() - used_minibatch_size + 1));
    std::uint32_t start_index_inside_sequence = (rand()%( /* If the memory is truncated for the training.. */
      environment->get_sequence_size() - used_sequence_truncation + 1 /* ..not all result output values are evaluated.. */
    )); /* ..only settings.get_memory_truncation(), starting at a random index inside bounds */
    return evaluate(
      sequence_start_index, used_minibatch_size,
      start_index_inside_sequence, used_sequence_truncation
    );
  }

  rafko_utilities::ConstVectorSubrange<> solve(
    const std::vector<double>& input,
    bool reset_neuron_data = false, std::uint32_t thread_index = 0
  ){
    return agent->solve(input, reset_neuron_data, thread_index);
  }

  void push_state(){
    environment->push_state();
  }

  void pop_state(){
    environment->pop_state();
  }

  constexpr rafko_mainframe::RafkoSettings& expose_settings(){
    return settings;
  }

  constexpr const rafko_net::RafkoNet& expose_network(){
    return network;
  }
  /* --- Methods taken from @RafkoContext --- */

private:
  rafko_mainframe::RafkoSettings settings;
  rafko_net::RafkoNet& network;
  std::unique_ptr<rafko_net::Solution> network_solution;
  std::unique_ptr<rafko_net::SolutionSolver> agent;
  std::shared_ptr<rafko_gym::RafkoEnvironment> environment;
  std::shared_ptr<rafko_gym::RafkoObjective> objective;
  std::shared_ptr<rafko_gym::RafkoWeightUpdater> weight_updater;

  std::vector<std::vector<double>> neuron_outputs_to_evaluate; /* for each feature array inside each sequence inside each thread in one evaluation iteration */
  rafko_utilities::ThreadGroup execution_threads;

  std::uint32_t used_sequence_truncation;
  std::uint32_t used_minibatch_size;

  /**
   * @brief      Evaluate the given data set with the given parameters
   *
   * @param[in]  sequence_start             The starting sequence to be evaluated inside the @data_set
   * @param[in]  sequences_to_evaluate      The number of sequences to evaluate inside the @data_set
   * @param[in]  start_index_in_sequence    Parameter for sequence truncation: only update error value starting from this index in every sequence
   * @param[in]  sequence_tructaion         The number of labels to evaluate inside every evaluated sequence
   * @return     The resulting fitness
   */
  double evaluate(std::uint32_t sequence_start, std::uint32_t sequences_to_evaluate, std::uint32_t start_index_in_sequence, std::uint32_t sequence_tructaion);


  double error_post_process(double raw_error, std::uint32_t labels_evaluated);
};

} /* namespace rafko_mainframe */

#endif /* RAFKO_CPU_CONTEXT_H */

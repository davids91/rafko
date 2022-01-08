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

  void set_environment(std::unique_ptr<rafko_gym::RafkoEnvironment> environment_){
    environment.reset();
    environment = std::move(environment_);
  }
  const rafko_gym::RafkoEnvironment& get_environment(){
    return *environment;
  }
  void set_objective(std::unique_ptr<rafko_gym::RafkoObjective> objective_){
    objective.reset();
    objective = std::move(objective_);
  }
  const rafko_gym::RafkoObjective& get_objective(){
    return *objective;
  }
  void set_weight_updater(std::unique_ptr<rafko_gym::RafkoWeightUpdater> weight_updater_){
    weight_updater.reset();
    weight_updater = std::move(weight_updater_);
  }

  rafko_gym::RafkoWeightUpdater& expose_weight_updater(){
    return *weight_updater;
  };

  sdouble32 full_evaluation(){
    evaluate(0u, environment->get_number_of_sequences(), 0u, environment->get_sequence_size());
    loops_unchecked = 0u;
    return -objective->get_feature_fitness();
  }

  sdouble32 stochastic_evaluation(uint32 seed = 0u){
    if(0u < seed)srand(seed);
    check();
    uint32 sequence_start_index = (rand()%(environment->get_number_of_sequences() - settings.get_minibatch_size() + 1));
    uint32 start_index_inside_sequence = (rand()%( /* If the memory is truncated for the training.. */
      environment->get_sequence_size() - settings.get_memory_truncation() + 1 /* ..not all result output values are evaluated.. */
    )); /* ..only settings.get_memory_truncation(), starting at a random index inside bounds */
    evaluate(sequence_start_index, settings.get_minibatch_size(), start_index_inside_sequence, used_sequence_truncation);
    ++loops_unchecked; ++iteration;
    return -objective->get_feature_fitness();
  }

  void push_state(){
    environment->push_state();
    objective->push_state();
  }

  void pop_state(){
    environment->pop_state();
    objective->pop_state();
  }

  /**
   * @brief      Checks if the environment stored state changed enough to be re-evaluated fully, and if it does, it re-evaluates it
   *
   * @param[in]      agent    The actor to be evaluated in the current environment
   */
  void check(){
    if(
      (loops_unchecked >= settings.get_tolerance_loop_value())
      ||(loops_unchecked > (objective->get_feature_fitness()/settings.get_learning_rate()))
    ){ /* calculate the error value for the agent in this environment */
      full_evaluation();
      loops_unchecked = 0;
    }
  }

  sdouble32 get_current_fitness(){
    return -objective->get_feature_fitness();
  }

  rafko_mainframe::RafkoSettings& expose_settings(){
    return settings;
  }

  rafko_net::RafkoNet& expose_network(){
    return network;
  }

private:
  class RafkoDummyObjective : public rafko_gym::RafkoObjective{
  public:
    ~RafkoDummyObjective() = default;
    void set_features_for_sequences(
      const std::vector<std::vector<sdouble32>>& neuron_data, uint32 neuron_buffer_index,
      uint32 sequence_start_index, uint32 sequences_to_evaluate,
      uint32 start_index_in_sequence, uint32 sequence_truncation
    ){
      parameter_not_used(neuron_data);
      parameter_not_used(neuron_buffer_index);
      parameter_not_used(sequence_start_index);
      parameter_not_used(sequences_to_evaluate);
      parameter_not_used(start_index_in_sequence);
      parameter_not_used(sequence_truncation);
    }
    void set_features_for_sequences(
      const std::vector<std::vector<sdouble32>>& neuron_data, uint32 neuron_buffer_index,
      uint32 sequence_start_index, uint32 sequences_to_evaluate,
      uint32 start_index_in_sequence, uint32 sequence_truncation,
      std::vector<sdouble32>& tmp_data
    ){
      parameter_not_used(neuron_data);
      parameter_not_used(neuron_buffer_index);
      parameter_not_used(sequence_start_index);
      parameter_not_used(sequences_to_evaluate);
      parameter_not_used(start_index_in_sequence);
      parameter_not_used(sequence_truncation);
      parameter_not_used(tmp_data);
    }
    const std::vector<sdouble32>& get_feature_fitness_vector()const{ return dummy; }
    sdouble32 get_feature_fitness()const{ return double_literal(0.0); };
    void reset_errors(){ };
    void push_state(){ };
    void pop_state(){ };
    void expose_to_multithreading(){ };
    void conceal_from_multithreading(){ };
  private:
    std::vector<sdouble32> dummy;
  };

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

  uint32 iteration = 1;
  uint32 loops_unchecked;
  uint32 used_sequence_truncation;

  /**
   * @brief      Evaluate the given data set with the given parameters
   *
   * @param[in]  sequence_start             The starting sequence to be evaluated inside the @data_set
   * @param[in]  sequences_to_evaluate      The number of sequences to evaluate inside the @data_set
   * @param[in]  start_index_in_sequence    Parameter for sequence truncation: only update error value starting from this index in every sequence
   * @param[in]  sequence_tructaion         The number of labels to evaluate inside every evaluated sequence
   */
  void evaluate(uint32 sequence_start, uint32 sequences_to_evaluate, uint32 start_index_in_sequence, uint32 sequence_tructaion);

  /**
   * @brief      Evaluate a single sequence. The evaluated sequences lies under @sequence_index + @thread_index
   *              as inside a multi-threaded evaluation, one thread is to evaluate one sequence
   *
   * @param[in]  sequence_index      The sequence to be evaluated inside the @data_set
   * @param[in]  thread_index        The index of the thread the function is used with inside @solve_threads
   */
  void evaluate_single_sequence(uint32 sequence_index, uint32 thread_index);
};

} /* namespace rafko_mainframe */

#endif /* RAFKO_CPU_CONTEXT_H */

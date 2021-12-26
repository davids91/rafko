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

#ifndef RAFKO_ENVIRONMENT_DATA_SET_H
#define RAFKO_ENVIRONMENT_DATA_SET_H

#include "rafko_global.h"

#include <vector>
#include <functional>

#include "rafko_utilities/services/thread_group.h"

#include "rafko_gym/models/data_aggregate.h"
#include "rafko_gym/services/rafko_environment.h"
#include "rafko_gym/services/rafko_agent.h"

namespace rafko_gym{

/**
 * @brief      A class representing an environment using a train and test set
 */
class RAFKO_FULL_EXPORT RafkoEnvironmentDataSet : public RafkoEnvironment{
public:
  RafkoEnvironmentDataSet(rafko_mainframe::RafkoServiceContext& service_context_, DataAggregate& train_set_, DataAggregate& test_set_);

  void install_agent(RafkoAgent& agent){
    agents.push_back(agent);
  }

  sdouble32 full_evaluation(){
    evaluate(agents.back(), train_set, 0u, train_set.get_number_of_sequences(), 0u, train_set.get_sequence_size());
    evaluate(agents.back(), test_set, 0u, test_set.get_number_of_sequences(), 0u, train_set.get_sequence_size());
    loops_unchecked = 0u;
    return -train_set.get_error_sum();
  }

  sdouble32 stochastic_evaluation(uint32 seed = 0){
    if(0 < seed)srand(seed);
    check();
    uint32 sequence_start_index = (rand()%(train_set.get_number_of_sequences() - service_context.get_minibatch_size() + 1));
    uint32 start_index_inside_sequence = (rand()%( /* If the memory is truncated for the training.. */
      train_set.get_sequence_size() - service_context.get_memory_truncation() + 1 /* ..not all result output values are evaluated.. */
    )); /* ..only service_context.get_memory_truncation(), starting at a random index inside bounds */
    evaluate(agents.back(), train_set, sequence_start_index, service_context.get_minibatch_size(), start_index_inside_sequence, used_sequence_truncation);
    ++loops_unchecked; ++iteration;
    return -train_set.get_error_sum();
  }

  void push_state(){
    train_set.push_state();
    test_set.push_state();
  }

  void pop_state(){
    train_set.pop_state();
    test_set.pop_state();
  }

  sdouble32 get_training_fitness(){
    return -train_set.get_error_avg();
  }

  sdouble32 get_testing_fitness(){
    return -test_set.get_error_avg();
  }

  /**
   * @brief      Checks if the environment stored state changed enough to be re-evaluated fully, and if it does, it re-evaluates it
   *
   * @param[in]      agent    The actor to be evaluated in the current environment
   */
  void check(){
    if(
      (loops_unchecked >= service_context.get_tolerance_loop_value())
      ||(loops_unchecked > (train_set.get_error_sum()/service_context.get_learning_rate()))
      ||(loops_unchecked > (test_set.get_error_sum()/service_context.get_learning_rate()))
    ){ /* calculate the error value for the agent in this environment */
      full_evaluation();
      loops_unchecked = 0;
    }
  }

  ~RafkoEnvironmentDataSet() = default;

private:
  rafko_mainframe::RafkoServiceContext& service_context;
  std::vector<std::reference_wrapper<RafkoAgent>> agents;
  DataAggregate& train_set;
  DataAggregate& test_set;
  std::vector<std::vector<sdouble32>> neuron_outputs_to_evaluate; /* for each feature array inside each sequence inside each thread in one evaluation iteration */
  rafko_utilities::ThreadGroup execution_threads;

  uint32 iteration = 1;
  uint32 loops_unchecked;
  uint32 used_sequence_truncation;

  /**
   * @brief      Evaluate the given data set with the given parameters
   *
   * @param      agent                   The agent to evaluate
   * @param      data_set                   The data set to evaluate the @agent on
   * @param[in]  sequence_start             The starting sequence to be evaluated inside the @data_set
   * @param[in]  sequences_to_evaluate      The number of sequences to evaluate inside the @data_set
   * @param[in]  start_index_in_sequence    Parameter for sequence truncation: only update error value starting from this index in every sequence
   * @param[in]  sequence_tructaion         The number of labels to evaluate inside every evaluated sequence
   */
  void evaluate(
    RafkoAgent& agent, DataAggregate& data_set, uint32 sequence_start, uint32 sequences_to_evaluate,
    uint32 start_index_in_sequence, uint32 sequence_tructaion
  );

  /**
   * @brief      Evaluate a single sequence. The evaluated sequences lies under @sequence_index + @thread_index
   *              as inside a multi-threaded evaluation, one thread is to evaluate one sequence
   *
   * @param      agent               The agent to evaluate
   * @param      data_set            The data set to evaluate the @agent on
   * @param[in]  sequence_index      The sequence to be evaluated inside the @data_set
   * @param[in]  thread_index        The index of the thread the function is used with inside @solve_threads
   */
  void evaluate_single_sequence(RafkoAgent& agent, DataAggregate& data_set, uint32 sequence_index, uint32 thread_index);
};

} /* namespace rafko_gym */
#endif /* RAFKO_ENVIRONMENT_DATA_SET_H */

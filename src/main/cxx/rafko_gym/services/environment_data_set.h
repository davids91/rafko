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

#ifndef ENVIRONMENT_DATA_SET_H
#define ENVIRONMENT_DATA_SET_H

#include "rafko_global.h"

#include <vector>
#include <functional>

#include "rafko_utilities/services/thread_group.h"

#include "rafko_gym/models/data_aggregate.h"
#include "rafko_gym/services/environment.h"
#include "rafko_gym/services/agent.h"

namespace rafko_gym{

using std::vector;
using std::reference_wrapper;

/**
 * @brief      A class representing an environment using a train and test set
 */
class Environment_data_set : public Environment{
public:
  Environment_data_set(Service_context& service_context_, Data_aggregate& train_set_, Data_aggregate& test_set_);

  sdouble32 full_evaluation(Agent& agent){
    evaluate(agent, train_set, 0u, train_set.get_number_of_sequences(), 0u, train_set.get_sequence_size());
    evaluate(agent, test_set, 0u, test_set.get_number_of_sequences(), 0u, train_set.get_sequence_size());
    loops_unchecked = 0u;
    return -train_set.get_error_sum();
  }

  sdouble32 stochastic_evaluation(Agent& agent, uint32 seed = 0){
    if(0 < seed)srand(seed);
    check(agent);
    uint32 sequence_start_index = (rand()%(train_set.get_number_of_sequences() - service_context.get_minibatch_size() + 1));
    uint32 start_index_inside_sequence = (rand()%( /* If the memory is truncated for the training.. */
      train_set.get_sequence_size() - service_context.get_memory_truncation() + 1 /* ..not all result output values are evaluated.. */
    )); /* ..only service_context.get_memory_truncation(), starting at a random index inside bounds */
    evaluate(agent, train_set, sequence_start_index, service_context.get_minibatch_size(), start_index_inside_sequence, service_context.get_memory_truncation());
    ++loops_unchecked; ++iteration;
    return -train_set.get_error_sum();
  }

  void push_state(void){
    train_set.push_state();
    test_set.push_state();
  }

  void pop_state(void){
    train_set.pop_state();
    test_set.pop_state();
  }

  sdouble32 get_training_fitness(void){
    return -train_set.get_error_avg();
  }

  sdouble32 get_testing_fitness(void){
    return -test_set.get_error_avg();
  }

  /**
   * @brief      Checks if the environment stored state changed enough to be re-evaluated fully, and if it does, it re-evaluates it
   *
   * @param[in]      agent    The actor to be evaluated in the current environment
   */
  void check(Agent& agent){
    if(
      (loops_unchecked >= service_context.get_tolerance_loop_value())
      ||(loops_unchecked > (train_set.get_error_sum()/service_context.get_learning_rate()))
      ||(loops_unchecked > (test_set.get_error_sum()/service_context.get_learning_rate()))
    ){ /* calculate the error value for the agent in this environment */
      full_evaluation(agent);
      loops_unchecked = 0;
    }
  }

  ~Environment_data_set(void) = default;

private:
  Service_context& service_context;
  Data_aggregate& train_set;
  Data_aggregate& test_set;
  vector<vector<sdouble32>> neuron_outputs_to_evaluate; /* for each feature array inside each sequence inside each thread in one evaluation iteration */
  ThreadGroup execution_threads;

  uint32 iteration = 1;
  uint32 loops_unchecked;
  uint32 sequence_truncation;

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
    Agent& agent, Data_aggregate& data_set, uint32 sequence_start, uint32 sequences_to_evaluate,
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
  void evaluate_single_sequence(Agent& agent, Data_aggregate& data_set, uint32 sequence_index, uint32 thread_index);
};

} /* namespace rafko_gym */
#endif /* ENVIRONMENT_DATA_SET_H */

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

#include "sparse_net_library/services/random_attention_brain.h"

#include <cmath>
#include <stdexcept>

#include "sparse_net_library/services/solution_builder.h"

namespace sparse_net_library{

using std::lock_guard;

Random_attention_brain::Random_attention_brain(SparseNet& neural_network, Data_aggregate& training_set_, Service_context& service_context)
:  net(neural_network)
,  context(service_context)
,  net_solution(Solution_builder(context).build(net))
,  solve_threads()
,  solvers()
,  weight_updater(net,context)
,  training_set(training_set_)
,  memory_truncation(std::min(context.get_memory_truncation(), training_set.get_sequence_size()))
,  weightxp_space()
,  dataset_mutex()
{
  for(sint32 weight_index = 0; weight_index < net.weight_table_size(); ++weight_index){
    weightxp_space.push_back(
      Weight_experience_space(double_literal(-1.0), double_literal(1.0), double_literal(0.2f))
    );
    net.set_weight_table(weight_index,weightxp_space[weight_index].get_best_weight());
  }
  weight_updater.update_solution_with_weights(*net_solution);
  solve_threads.reserve(context.get_max_solve_threads());
  for(uint32 threads = 0; threads < context.get_max_solve_threads(); ++threads){
    solvers.push_back(std::make_unique<Solution_solver>(*net_solution, service_context, training_set.get_sequence_size()));
  }
  evaluate(0,training_set.get_number_of_sequences(),0,training_set.get_sequence_size());
}

void Random_attention_brain::step(void){
  /* Choose a weight to examine */
  uint32 weight_index = rand()%(net.weight_table_size());

  /* Choose a random sample to evaluate performance on */
  uint32 sample_index = rand()%(training_set.get_number_of_sequences());
  uint32 start_index_inside_sequence = (rand()%( /* If the memory is truncated for the training */
    training_set.get_sequence_size() - memory_truncation + 1 /* not all result output values are evaluated */
  )); /* only context.get_memory_truncation(), starting at a random index inside bounds */
  uint32 index_inside_raw_labels = (sample_index * training_set.get_sequence_size()) + start_index_inside_sequence;

  evaluate(
    (sample_index * training_set.get_sequence_size()),training_set.get_sequence_size(),
    start_index_inside_sequence, memory_truncation
  );

  /* Read out the collected error from the training set */
  sdouble32 error_value = 0;
  for(uint32 labels_index = 0; labels_index < memory_truncation; ++labels_index){
    error_value += training_set.get_error(index_inside_raw_labels);
    ++index_inside_raw_labels;
  }

  /* Add error value as experience for the selected weight and update weights for the network */
  /*if(double_literal(0.0) == error_value)
    error_value = double_literal(-1.0); *//*!Note: "Reward" for being correct - seems to make training unstable */
  net.set_weight_table(weight_index, weightxp_space[weight_index].add_experience(-error_value));
  weight_updater.update_solution_with_weights(*net_solution);

  /* If error appears to be quite small, evaluate the whole training dataset */
  if(context.get_step_size() >= error_value)
    evaluate(0,training_set.get_number_of_sequences(),0,training_set.get_sequence_size());
}

void Random_attention_brain::evaluate(
  uint32 label_start_index, uint32 labels_to_eval,
  uint32 start_index_inside_sequence, uint32 sequence_truncation
){
  uint32 sequence_index = label_start_index;
  const uint32 sequences_to_evaluate = std::min(training_set.get_number_of_sequences(), labels_to_eval);
  const uint32 sequences_in_one_thread = 1 + static_cast<uint32>(sequences_to_evaluate/context.get_max_solve_threads());

  for(uint32 thread_index = 0; ((thread_index < context.get_max_solve_threads())&&(training_set.get_number_of_sequences() > sequence_index)); ++thread_index){
    solve_threads.push_back(thread( /* As long as there are threads to open for remaining weights, open threads */
      &Random_attention_brain::evaluate_thread, this, thread_index, sequence_index,
      std::min(sequences_in_one_thread, (training_set.get_number_of_sequences() - sequence_index)),
      start_index_inside_sequence, sequence_truncation
    ));
    sequence_index += sequences_in_one_thread;
  }

  while(0 < solve_threads.size()){
    if(solve_threads.back().joinable()){
      solve_threads.back().join();
      solve_threads.pop_back();
    }
  }
}

void Random_attention_brain::evaluate_thread(
  uint32 solve_thread_index, uint32 sequence_start_index, uint32 sequences_to_evaluate,
  uint32 start_index_inside_sequence, uint32 sequence_truncation
){
  uint32 raw_inputs_index;
  uint32 raw_label_index;

  if(training_set.get_number_of_sequences() < (sequence_start_index + sequences_to_evaluate))
    throw std::runtime_error("Sequence range out of bounds!");

  for(uint32 sequence_index = sequence_start_index; sequence_index < (sequence_start_index + sequences_to_evaluate); ++sequence_index){
    raw_inputs_index = sequence_index * (training_set.get_sequence_size() + training_set.get_prefill_inputs_number());
    raw_label_index = sequence_index * training_set.get_sequence_size();

    /* Prefill network with the initial inputs of that sample */
    solvers[solve_thread_index]->reset();
    for(uint32 prefill_iterator = 0; prefill_iterator < training_set.get_prefill_inputs_number(); ++prefill_iterator){
      solvers[solve_thread_index]->solve(training_set.get_input_sample(raw_inputs_index));
      ++raw_inputs_index;
    }

    /* Evaluate the current sequence step by step */
    for(uint32 sequence_iterator = 0; sequence_iterator < training_set.get_sequence_size(); ++sequence_iterator){
      solvers[solve_thread_index]->solve(training_set.get_input_sample(raw_inputs_index)); /* Solve the network for the sampled labels input */
      ++raw_label_index;
      ++raw_inputs_index;
    }

    lock_guard<mutex> my_lock(dataset_mutex);
    training_set.set_features_for_labels(
      solvers[solve_thread_index]->get_neuron_memory().get_whole_buffer(), start_index_inside_sequence,
      (sequence_index * training_set.get_sequence_size()) + start_index_inside_sequence,
      memory_truncation /* To avoid vanishing gradients with sequential data, error calculation is truncated */
    ); /* Re-calculate error for part of the training set */
  }
}

const Weight_experience_space& Random_attention_brain::get_weight_experiences(uint32 weight_index) const{
  if(weightxp_space.size() > weight_index){
    return weightxp_space[weight_index];
  }else throw std::runtime_error("Weight index out of bounds while requesting experiences!");
}

} /* namespace sparse_net_library */
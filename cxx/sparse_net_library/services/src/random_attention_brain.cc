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

#include "sparse_net_library/services/solution_builder.h"

namespace sparse_net_library{

Random_attention_brain::Random_attention_brain(SparseNet& neural_network, Data_aggregate& training_set_, Service_context& service_context)
:  net(neural_network)
,  context(service_context)
,  net_solution(Solution_builder(context).build(net))
,  solver(*net_solution, service_context, training_set_.get_sequence_size())
,  weight_updater(net,context)
,  training_set(training_set_)
,  memory_truncation(std::min(context.get_memory_truncation(), training_set.get_sequence_size()))
,  weightxp_space()
{
  for(uint32 weight_index = 0; weight_index < net.weight_table_size(); ++weight_index){
    weightxp_space.push_back(
      Weight_experience_space(double_literal(-1.0), double_literal(1.0), double_literal(0.2f))
    );
    net.set_weight_table(weight_index,weightxp_space[weight_index].get_best_weight());
  }
  weight_updater.update_solution_with_weights(*net_solution);
}

void Random_attention_brain::step(void){
  /* Choose a weight to examine */
  uint32 weight_index = rand()%(net.weight_table_size());

  /* Choose a random sample to evaluate performance on */
  uint32 sample_index = rand()%(training_set.get_number_of_sequences());
  uint32 raw_inputs_index = sample_index * (training_set.get_sequence_size() + training_set.get_prefill_inputs_number());
  uint32 raw_label_index = sample_index * training_set.get_sequence_size();
  uint32 start_index_inside_sequence = (rand()%( /* If the memory is truncated for the training */
    training_set.get_sequence_size() - memory_truncation + 1 /* not all result output values are evaluated */
  )); /* only context.get_memory_truncation(), starting at a random index inside bounds */

  /* Prefill network with the initial inputs of that sample */
  solver.reset();
  for(uint32 prefill_iterator = 0; prefill_iterator < training_set.get_prefill_inputs_number(); ++prefill_iterator){
    solver.solve(training_set.get_input_sample(raw_inputs_index));
    ++raw_inputs_index;
  }

  /* Evaluate the current sequence step by step */
  for(uint32 sequence_iterator = 0; sequence_iterator < training_set.get_sequence_size(); ++sequence_iterator){
    solver.solve(training_set.get_input_sample(raw_inputs_index)); /* Solve the network for the sampled labels input */
    ++raw_label_index;
    ++raw_inputs_index;
  }

  training_set.set_features_for_labels(
    solver.get_neuron_memory().get_whole_buffer(), start_index_inside_sequence,
    (sample_index * training_set.get_sequence_size()) + start_index_inside_sequence,
    memory_truncation /* To avoid vanishing gradients with sequential data, error calculation is truncated */
  ); /* Re-calculate error for the training set */

  /* Read out the collected error from the training set */
  sdouble32 error_value = 0;
  raw_label_index = (sample_index * training_set.get_sequence_size()) + start_index_inside_sequence;
  for(uint32 labels_index = 0; labels_index < memory_truncation; ++labels_index){
    error_value += training_set.get_error(raw_label_index);
    ++raw_label_index;
  }

  /* Add error value as experience for the selected weight and update weights for the network */
  net.set_weight_table(weight_index, weightxp_space[weight_index].add_experience(-error_value));
  weight_updater.update_solution_with_weights(*net_solution);
}

} /* namespace sparse_net_library */
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
,  solvers()
,  weight_updater(net,context)
,  training_set(training_set_)
,  memory_truncation(std::min(context.get_memory_truncation(), training_set.get_sequence_size()))
,  weightxp_space()
,  iteration(1)
{
  for(sint32 weight_index = 0; weight_index < net.weight_table_size(); ++weight_index){
    weightxp_space.push_back(
      Weight_experience_space(double_literal(-1.0), double_literal(1.0), double_literal(0.2f))
    );
    net.set_weight_table(weight_index,weightxp_space[weight_index].get_best_weight());
  }
  weight_updater.update_solution_with_weights(*net_solution);
  for(uint32 threads = 0; threads < context.get_max_solve_threads(); ++threads){
    solvers.push_back(std::make_unique<Solution_solver>(*net_solution, service_context));
  }
  training_set.set_features_for_labels(solvers,0,training_set.get_number_of_sequences(),0,training_set.get_sequence_size());
}

void Random_attention_brain::step(void){
  /* Choose a weight to examine */
  uint32 weight_index = rand()%(net.weight_table_size());

  /* Choose a random sample to evaluate performance on */
  uint32 sample_index = rand()%(training_set.get_number_of_sequences());
  uint32 start_index_inside_sequence_start = (rand()%( /* If the memory is truncated for the training */
    training_set.get_sequence_size() - memory_truncation + 1 /* not all result output values are evaluated */
  )); /* only context.get_memory_truncation(), starting at a random index inside bounds */
  uint32 index_inside_raw_labels_start = (sample_index * training_set.get_sequence_size()) + start_index_inside_sequence_start;

  /* Get the current error value */
  sdouble32 error_value = 0;
  sdouble32 current_error_value;
  uint32 start_index_inside_sequence = start_index_inside_sequence_start;
  uint32 index_inside_raw_labels = index_inside_raw_labels_start;
  training_set.set_features_for_labels(
    solvers, (sample_index * training_set.get_sequence_size()),
    training_set.get_sequence_size(), start_index_inside_sequence, memory_truncation
  );
  for(uint32 labels_index = 0; labels_index < memory_truncation; ++labels_index){
    error_value += training_set.get_error(index_inside_raw_labels);
    ++index_inside_raw_labels;
  }
  current_error_value = error_value;

  /* Get the error value of the left neighbour */
  sdouble32 context_avg_error;
  start_index_inside_sequence = start_index_inside_sequence_start;
  index_inside_raw_labels = index_inside_raw_labels_start;
  net.set_weight_table(weight_index, weightxp_space[weight_index].get_best_weight() - context.get_step_size());
  weight_updater.update_solution_with_weights(*net_solution);

  error_value = 0;
  training_set.set_features_for_labels(
    solvers, (sample_index * training_set.get_sequence_size()),
    training_set.get_sequence_size(), start_index_inside_sequence, memory_truncation
  );
  for(uint32 labels_index = 0; labels_index < memory_truncation; ++labels_index){
    error_value += training_set.get_error(index_inside_raw_labels);
    ++index_inside_raw_labels;
  }
  context_avg_error = error_value;

  /* Get the error value of the right neighbour */
  start_index_inside_sequence = start_index_inside_sequence_start;
  index_inside_raw_labels = index_inside_raw_labels_start;
  net.set_weight_table(weight_index, weightxp_space[weight_index].get_best_weight() + context.get_step_size());
  weight_updater.update_solution_with_weights(*net_solution);
  error_value = 0;
  training_set.set_features_for_labels(
    solvers, (sample_index * training_set.get_sequence_size()),
    training_set.get_sequence_size(), start_index_inside_sequence, memory_truncation
  );
  for(uint32 labels_index = 0; labels_index < memory_truncation; ++labels_index){
    error_value += training_set.get_error(index_inside_raw_labels);
    ++index_inside_raw_labels;
  }
  context_avg_error += error_value;
  context_avg_error /= double_literal(2.0);

  /* add the impulse, update the weight */
  net.set_weight_table(weight_index, weightxp_space[weight_index].add_experience(
    -context_avg_error/std::abs(current_error_value) * std::min(double_literal(1.0),std::log(iteration)/double_literal(3.0)))
  );
  weight_updater.update_solution_with_weights(*net_solution);


  /* If error appears to be quite small, evaluate the whole training dataset */
  if(context.get_step_size() >= error_value)
    training_set.set_features_for_labels(solvers,0,training_set.get_number_of_sequences(),0,training_set.get_sequence_size());

  ++iteration;
}

const Weight_experience_space& Random_attention_brain::get_weight_experiences(uint32 weight_index) const{
  if(weightxp_space.size() > weight_index){
    return weightxp_space[weight_index];
  }else throw std::runtime_error("Weight index out of bounds while requesting experiences!");
}

} /* namespace sparse_net_library */

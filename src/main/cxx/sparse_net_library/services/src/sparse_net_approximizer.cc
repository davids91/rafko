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

#include "sparse_net_library/services/sparse_net_approximizer.h"

#include <algorithm>

#include "gen/common.pb.h"

#include "sparse_net_library/services/synapse_iterator.h"
#include "sparse_net_library/services/solution_builder.h"
#include "sparse_net_library/services/updater_factory.h"

namespace sparse_net_library{


using std::make_unique;
using std::lock_guard;
using std::ref;

Sparse_net_approximizer::Sparse_net_approximizer(
  SparseNet& neural_network, Data_aggregate& train_set_, Data_aggregate& test_set_,
  weight_updaters weight_updater_, Service_context& service_context
): net(neural_network)
,  context(service_context)
,  train_set(train_set_)
,  test_set(test_set_)
,  net_solution(Solution_builder(context).build(net))
,  solver(Solution_solver::Builder(*net_solution, service_context).build())
,  neuron_value_buffers( context.get_max_processing_threads(), DataRingbuffer( net_solution->network_memory_length(), net_solution->neuron_number()) )
,  neuron_outputs_to_evaluate((context.get_max_processing_threads() * train_set.get_sequence_size()), vector<sdouble32>(net_solution->neuron_number()))
,  solve_threads()
,  instance_data_pool(context.get_max_processing_threads(), solver->get_required_temp_data_size())
,  used_data_pools()
,  gradient_fragment()
,  iteration(1)
,  loops_unchecked(context.get_insignificant_changes())
,  sequence_truncation(min(context.get_memory_truncation(), train_set.get_sequence_size()))
,  last_applied_direction(net.weight_table_size())
{
  (void)context.set_minibatch_size(max(1u,min(
    train_set.get_number_of_sequences(),context.get_minibatch_size()
  )));
  (void)context.set_memory_truncation(max(1u,min(
    train_set.get_sequence_size(), context.get_memory_truncation()
  )));
  if(train_set.get_feature_size() != solver->get_solution().output_neuron_number())
    throw std::runtime_error("Network output size doesn't match size of provided labels!");
  weight_updater = Updater_factory::build_weight_updater(net,weight_updater_,context);
  solve_threads.reserve(context.get_max_processing_threads());

  /* A temporary buffer is allocated at initialization for every thread inside the approximizer * every thread inside the solver */
  for(uint32 process_thread_iterator = 0; process_thread_iterator < (context.get_max_processing_threads() * context.get_max_solve_threads()); ++process_thread_iterator)
    used_data_pools.push_back(instance_data_pool.reserve_buffer(solver->get_required_temp_data_size()));

  /* Plus 1 for the data set evaluation */
  used_data_pools.push_back(instance_data_pool.reserve_buffer(train_set.get_number_of_label_samples()));

  evaluate();
}

void Sparse_net_approximizer::evaluate(void){
  evaluate(train_set, 0, train_set.get_number_of_sequences(), 0, train_set.get_sequence_size());
  evaluate(test_set, 0, test_set.get_number_of_sequences(), 0, train_set.get_sequence_size());
}

void Sparse_net_approximizer::evaluate(Data_aggregate& data_set, uint32 sequence_start, uint32 sequences_to_evaluate, uint32 start_index_in_sequence, uint32 sequence_tructaion){
  if(data_set.get_number_of_sequences() < (sequence_start + sequences_to_evaluate))
    throw std::runtime_error("Sequence interval out of bounds!");
  data_set.expose_to_multithreading();

  for(uint32 sequence_index = sequence_start; sequence_index < (sequence_start + sequences_to_evaluate); sequence_index += context.get_max_processing_threads()){ /* one evaluation iteration */
    for(uint32 thread_index = 0; ((thread_index < context.get_max_processing_threads())&&((sequence_start + sequences_to_evaluate) > (sequence_index + thread_index))); ++thread_index){
      solve_threads.push_back(thread(&Sparse_net_approximizer::evaluate_single_sequence, this, ref(data_set), sequence_index, thread_index));
    }
    while(0 < solve_threads.size()){
      if(solve_threads.back().joinable()){
        solve_threads.back().join();
        solve_threads.pop_back();
      }
    }

    data_set.set_features_for_sequences( /* Upload results to the data set */
      neuron_outputs_to_evaluate, 0u,
      sequence_index, min(((sequence_start + sequences_to_evaluate) - (sequence_index)),static_cast<uint32>(context.get_max_processing_threads())),
      start_index_in_sequence, sequence_truncation,
      used_data_pools[(context.get_max_solve_threads() * context.get_max_processing_threads())]
    );
  } /* for(sequence_index: sequence_start --> (sequence start + sequences_to_evaluate)) */
  data_set.conceal_from_multithreading();
}

void Sparse_net_approximizer::evaluate_single_sequence(Data_aggregate& data_set, uint32 sequence_index, uint32 thread_index){
  /* Solve the sequence under sequence_index + thread_index */
  uint32 raw_label_index = sequence_index + thread_index;
  uint32 raw_inputs_index = raw_label_index * (data_set.get_sequence_size() + data_set.get_prefill_inputs_number()); /* calculate the raw input arrays index */
  raw_label_index *= data_set.get_sequence_size(); /* calculate the raw labels array index */

  /* Evaluate the current sequence step by step */
  neuron_value_buffers[thread_index].reset();
  for(uint32 prefill_iterator = 0; prefill_iterator < data_set.get_prefill_inputs_number(); ++prefill_iterator){
    solver->solve(
      data_set.get_input_sample(raw_inputs_index), neuron_value_buffers[thread_index],
      used_data_pools, (thread_index * context.get_max_solve_threads())
    ); /* step is included in @solve */
    ++raw_inputs_index;
  }

  /* Solve the data and store the result after the inital "prefill" */
  for(uint32 sequence_iterator = 0; sequence_iterator < data_set.get_sequence_size(); ++sequence_iterator){
    solver->solve(
      data_set.get_input_sample(raw_inputs_index), neuron_value_buffers[thread_index],
      used_data_pools, (thread_index * context.get_max_solve_threads())
    );
    std::copy( /* copy the result to the eval array */
      neuron_value_buffers[thread_index].get_element(0).begin(),neuron_value_buffers[thread_index].get_element(0).end(),
      neuron_outputs_to_evaluate[(thread_index * data_set.get_sequence_size()) + sequence_iterator].begin()
    );
    ++raw_label_index;
    ++raw_inputs_index;
  }
}

void Sparse_net_approximizer::collect_approximates_from_weight_gradients(void){
  vector<sdouble32> weight_gradients(net.weight_table_size(),double_literal(0.0));
  uint32 index_of_biggest = 0;
  sdouble32 sum_gradient = double_literal(0.0);
  sdouble32 gradient_overview = get_gradient_for_all_weights() * context.get_step_size();

  for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
    weight_gradients[weight_index] += get_single_weight_gradient(weight_index);
    sum_gradient += std::pow(weight_gradients[weight_index],double_literal(2.0));
    if(std::abs(weight_gradients[index_of_biggest]) < std::abs(weight_gradients[weight_index]))
      index_of_biggest = weight_index;
  }
  sum_gradient = std::sqrt(sum_gradient);

  convert_direction_to_gradient(last_applied_direction,false); /* check the last applied direction */
  for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
    weight_gradients[weight_index] = (
      ( /* Gradients normalized by the biggest value */
        (weight_gradients[weight_index] / std::abs(weight_gradients[index_of_biggest]))
        + gradient_overview /* plus the overview gradient */
      ) / (double_literal(1.0) + std::abs(gradient_overview)) /* normalized by the extended value */
    ); /*!Note: the biggest value in the weight gradients should be at most 1.0 after normalization,
        * so dividing by 1.0 + gradient_overview should normalize the offseted gradients
        */
    weight_gradients[weight_index] *= context.get_step_size();
  }

  convert_direction_to_gradient(weight_gradients,true);
}

void Sparse_net_approximizer::collect_approximates_from_random_direction(void){
  vector<sdouble32> direction(net.weight_table_size());
  sdouble32 gradient_overview = get_gradient_for_all_weights() * context.get_step_size();
  std::generate_n(direction.begin(),net.weight_table_size(),[=](){
    return(
      (
        (static_cast<sdouble32>(rand()%201) / double_literal(200.0)) - double_literal(1.0) + gradient_overview
      ) * context.get_step_size()
    );
  });
  convert_direction_to_gradient(direction,true);
}

void Sparse_net_approximizer::convert_direction_to_gradient(vector<sdouble32>& direction, bool save_to_fragment){
  if(net.weight_table_size() == static_cast<sint32>(direction.size())){
    check();

    sdouble32 dampening_value;
    sdouble32 error_negative_direction;
    sdouble32 error_positive_direction;
    vector<sdouble32> weight_gradients(net.weight_table_size(),double_literal(0.0));
    vector<sdouble32> weight_steps(net.weight_table_size(),double_literal(0.0));
    sdouble32 weight_epsilon = double_literal(0.0);
    uint32 sequence_start_index = (rand()%(
      train_set.get_number_of_sequences() - context.get_minibatch_size() + 1
    ));
    uint32 start_index_inside_sequence = (rand()%( /* If the memory is truncated for the training */
      train_set.get_sequence_size() - context.get_memory_truncation() + 1 /* not all result output values are evaluated */
    )); /* only context.get_memory_truncation(), starting at a random index inside bounds */
    train_set.push_state();

    /* apply the direction on which network approximation shall be done */
    for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
      weight_steps[weight_index] = direction[weight_index];
      weight_epsilon += std::pow(weight_steps[weight_index], double_literal(2.0));
      net.set_weight_table(weight_index, (net.weight_table(weight_index) - weight_steps[weight_index]) );
    };
    weight_updater->update_solution_with_weights(*net_solution);
    weight_epsilon = std::sqrt(weight_epsilon) * double_literal(2.0);

    /* see the error values at the negative end of the current direction */
    evaluate(train_set, sequence_start_index, context.get_minibatch_size(), start_index_inside_sequence, context.get_memory_truncation());
    error_negative_direction = train_set.get_error_sum();

    /* see the error values at the positive end of the current direction */
    for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index)
      net.set_weight_table(weight_index, (net.weight_table(weight_index) + (weight_steps[weight_index] * double_literal(2.0))) );
    weight_updater->update_solution_with_weights(*net_solution);
    evaluate(train_set, sequence_start_index, context.get_minibatch_size(), start_index_inside_sequence, context.get_memory_truncation());
    error_positive_direction = train_set.get_error_sum();

    /* Restore train set to previous error state, decide if dampening is needed */
    train_set.pop_state();
    if( /* In case the initial error is smaller, than the errors in either direction */
      (train_set.get_error_avg() < error_positive_direction)
      &&(train_set.get_error_avg() < error_negative_direction)
    )dampening_value = context.get_zetta(); /* decrease the amount to move the current net */
      else dampening_value = double_literal(1.0); /* reducing oscillation at lower error ranges */

    /* collect the fragment, revert weight changes */
    for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
      weight_gradients[weight_index] = (
        ( error_positive_direction - error_negative_direction ) / (weight_epsilon)
      );

      if(save_to_fragment)add_to_fragment( weight_index, (weight_gradients[weight_index] * weight_steps[weight_index] * dampening_value) );
      net.set_weight_table(weight_index, (net.weight_table(weight_index) - (weight_steps[weight_index])) );
    }
    weight_updater->update_solution_with_weights(*net_solution);
    ++loops_unchecked; ++iteration;
  }else throw std::runtime_error("Incompatible direction given to apprximate for!");
}

void Sparse_net_approximizer::collect_fragment(void){
  check();

  /* Approximate the gradient for every weight */
  vector<sdouble32> weight_gradients(net.weight_table_size(),double_literal(0.0));
  uint32 index_of_biggest = double_literal(0.0);
  sdouble32 average_gradient = double_literal(0.0);
  sdouble32 sum_gradient = double_literal(0.0);
  for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
    weight_gradients[weight_index] = get_single_weight_gradient(weight_index);
    average_gradient += weight_gradients[weight_index];
    sum_gradient += std::pow(weight_gradients[weight_index],double_literal(2.0));
    if(weight_gradients[index_of_biggest] < weight_gradients[weight_index])
      index_of_biggest = weight_index;
  }
  sum_gradient = std::sqrt(sum_gradient);
  average_gradient /= static_cast<sdouble32>(net.weight_table_size());
  add_to_fragment(index_of_biggest, (weight_gradients[index_of_biggest] / sum_gradient) );
  ++loops_unchecked; ++iteration;
}

sdouble32 Sparse_net_approximizer::get_single_weight_gradient(uint32 weight_index){
  sdouble32 gradient;
  const sdouble32 current_epsilon = context.get_sqrt_epsilon();
  const sdouble32 current_epsilon_double = current_epsilon * double_literal(2.0);

  /* Save error from the initial network, decide the minibatch and sequence truncation */
  train_set.push_state();
  uint32 sequence_start_index = (rand()%(
    train_set.get_number_of_sequences() - context.get_minibatch_size() + 1
  ));
  uint32 start_index_inside_sequence = (rand()%( /* If the memory is truncated for the training */
    train_set.get_sequence_size() - context.get_memory_truncation() + 1 /* not all result output values are evaluated */
  )); /* only context.get_memory_truncation(), starting at a random index inside bounds */

  /* Push it in one direction */
  net.set_weight_table(weight_index, (net.weight_table(weight_index) + current_epsilon) );
  weight_updater->update_solution_with_weights(*net_solution);

  /* Approximate the modified weights gradient */
  evaluate(train_set, sequence_start_index, context.get_minibatch_size(), start_index_inside_sequence, context.get_memory_truncation());
  gradient = train_set.get_error_sum();

  /* Push the selected weight in other direction */
  net.set_weight_table(weight_index, (net.weight_table(weight_index) - current_epsilon_double) );
  weight_updater->update_solution_with_weights(*net_solution);

  /* Approximate the newly modified weights gradient */
  evaluate(train_set, sequence_start_index, context.get_minibatch_size(), start_index_inside_sequence, context.get_memory_truncation());
  gradient = -(gradient - train_set.get_error_sum()) / (current_epsilon_double * context.get_minibatch_size());

  /* Revert weight modification and the error state with it */
  net.set_weight_table(weight_index, (net.weight_table(weight_index) + current_epsilon) );
  weight_updater->update_solution_with_weights(*net_solution);
  train_set.pop_state();

  return gradient;
}

sdouble32 Sparse_net_approximizer::get_gradient_for_all_weights(void){
  sdouble32 gradient;
  sdouble32 error_negative_direction;
  sdouble32 error_positive_direction;
  const sdouble32 current_epsilon = context.get_sqrt_epsilon();
  const sdouble32 current_epsilon_double = current_epsilon * double_literal(2.0);

  /* Save error from the initial network, decide the minibatch and sequence truncation */
  train_set.push_state();
  uint32 sequence_start_index = (rand()%(
    train_set.get_number_of_sequences() - context.get_minibatch_size() + 1
  ));
  uint32 start_index_inside_sequence = (rand()%( /* If the memory is truncated for the training */
    train_set.get_sequence_size() - context.get_memory_truncation() + 1 /* not all result output values are evaluated */
  )); /* only context.get_memory_truncation(), starting at a random index inside bounds */


  /* Push every weight in a positive epsilon direction */
  for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
    net.set_weight_table(weight_index, (net.weight_table(weight_index) + current_epsilon) );
  }
  weight_updater->update_solution_with_weights(*net_solution);

  /* Approximate the modified weights gradient */
  evaluate(train_set, sequence_start_index, context.get_minibatch_size(), start_index_inside_sequence, context.get_memory_truncation());
  error_positive_direction = train_set.get_error_sum();

  /* Push them in the other direction */
  for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
    net.set_weight_table(weight_index, (net.weight_table(weight_index) - current_epsilon_double) );
  }
  weight_updater->update_solution_with_weights(*net_solution);

  /* Approximate the newly modified weights gradient */
  evaluate(train_set, sequence_start_index, context.get_minibatch_size(), start_index_inside_sequence, context.get_memory_truncation());
  error_negative_direction = train_set.get_error_sum();

  gradient = -(error_positive_direction - error_negative_direction) / (current_epsilon_double * context.get_minibatch_size());

  /* Revert weight modifications and the error state with it */
  for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
    net.set_weight_table(weight_index, (net.weight_table(weight_index) + current_epsilon) );
  }
  weight_updater->update_solution_with_weights(*net_solution);
  train_set.pop_state();

  return gradient;
}

void Sparse_net_approximizer::add_to_fragment(uint32 weight_index, sdouble32 gradient_fragment_value){
  uint32 values_index = 0;
  uint32 values_index_target = gradient_fragment.values_size();
  uint32 weight_synapse_index_target = gradient_fragment.weight_synapses_size();
  Index_synapse_interval tmp_synapse_interval;

  for(uint32 weight_syn_index = 0; static_cast<sint32>(weight_syn_index) < gradient_fragment.weight_synapses_size(); ++weight_syn_index){
    if( /* If the weight synapse is at or in-between the first index before the start of the synapse.. */
      (
        ((0 < gradient_fragment.weight_synapses(weight_syn_index).starts())
        &&((gradient_fragment.weight_synapses(weight_syn_index).starts()-1) <= static_cast<sint32>(weight_index)))
        ||((0 == gradient_fragment.weight_synapses(weight_syn_index).starts())&&(0 <= weight_index))
      )&&( /* ..and the one after the last index */
        (gradient_fragment.weight_synapses(weight_syn_index).starts() + gradient_fragment.weight_synapses(weight_syn_index).interval_size())
        >= weight_index
      )
    ){ /* current weight synapse is a sitable target to place the current fragment in */
      weight_synapse_index_target = weight_syn_index;
      values_index_target = values_index;
      break; /* Found a suitable synapse, no need to continue */
    }
    values_index += gradient_fragment.weight_synapses(weight_syn_index).interval_size();
  } /* Go through the synapses saving the last place */
  if(
    (0 == gradient_fragment.weight_synapses_size())
    ||(static_cast<sint32>(weight_synapse_index_target) >= gradient_fragment.weight_synapses_size())
    ||(static_cast<sint32>(values_index_target) >= gradient_fragment.values_size())
  ){
    gradient_fragment.add_values(gradient_fragment_value);
    tmp_synapse_interval.set_interval_size(1);
    tmp_synapse_interval.set_starts(weight_index);
    *gradient_fragment.add_weight_synapses() = tmp_synapse_interval;
  }else{
    const uint32 synapse_starts = gradient_fragment.weight_synapses(weight_synapse_index_target).starts();
    const uint32 synapse_size = gradient_fragment.weight_synapses(weight_synapse_index_target).interval_size();
    const uint32 synapse_ends = synapse_starts + synapse_size;

    if(
      (0 < synapse_starts) /* Synapse doesn't start at 0 */
      &&((synapse_starts-1) == weight_index) /* And the weight index points to the first index before the synapse */
    ){
      gradient_fragment.mutable_weight_synapses(weight_synapse_index_target)->set_interval_size(synapse_size + 1);
      gradient_fragment.mutable_weight_synapses(weight_synapse_index_target)->set_starts(synapse_starts - 1);
      insert_element_at_position(*gradient_fragment.mutable_values(),gradient_fragment_value,values_index_target);
    }else if(
      (synapse_starts <= weight_index)
      &&(synapse_ends > weight_index)
    ){ /* the index is inside the synapse */
      gradient_fragment.set_values(
        values_index_target + weight_index - synapse_starts,
        gradient_fragment.values(values_index_target + weight_index - synapse_starts) + gradient_fragment_value
      );
    }else{ /* The index is the first index after the synapse */
      gradient_fragment.mutable_weight_synapses(weight_synapse_index_target)->set_interval_size(synapse_size + 1);
      insert_element_at_position(
        *gradient_fragment.mutable_values(), gradient_fragment_value, (values_index_target + synapse_size)
      );
    }
  }
}

void Sparse_net_approximizer::apply_fragment(void){
  uint32 fragment_value_index = 0;
  std::fill(last_applied_direction.begin(),last_applied_direction.end(),double_literal(0.0));

  if(weight_updater->is_finished())weight_updater->start();
  if(1 == gradient_fragment.weight_synapses_size()){
    std::copy(
      gradient_fragment.values().begin(),
      gradient_fragment.values().begin() + gradient_fragment.weight_synapses(0).interval_size(),
      last_applied_direction.begin() + gradient_fragment.weight_synapses(0).starts()
    );
  }else{
    Synapse_iterator<>::iterate(gradient_fragment.weight_synapses(), [&](
      Index_synapse_interval interval, sint32 weight_index
    ){
      last_applied_direction[weight_index] += gradient_fragment.values(fragment_value_index);
      ++fragment_value_index;
    });
  }

  weight_updater->iterate(last_applied_direction, *net_solution);
  gradient_fragment = Gradient_fragment();
  loops_unchecked = context.get_insignificant_changes() + 1;
}

} /* namespace sparse_net_library */

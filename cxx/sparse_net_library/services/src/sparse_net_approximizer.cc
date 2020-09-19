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

#include "gen/common.pb.h"

#include "sparse_net_library/services/synapse_iterator.h"
#include "sparse_net_library/services/solution_builder.h"
#include "sparse_net_library/services/updater_factory.h"

namespace sparse_net_library{

using std::lock_guard;
using std::ref;

Sparse_net_approximizer::Sparse_net_approximizer(
  SparseNet& neural_network, Data_aggregate& train_set_, Data_aggregate& test_set_,
  weight_updaters weight_updater_, Service_context& service_context
): net(neural_network)
,  context(service_context)
,  net_solution(Solution_builder(context).build(net))
,  solvers()
,  train_set(train_set_)
,  test_set(test_set_)
,  gradient_fragment()
,  loops_unchecked(context.get_insignificant_iteration_count())
,  sequence_truncation(min(context.get_memory_truncation(), train_set.get_sequence_size()))
,  solve_threads()
,  process_threads(context.get_max_solve_threads()) /* One queue for every solve thread */
,  dataset_mutex()
{
  (void)context.set_minibatch_size(max(1u,min(
    train_set.get_number_of_sequences(),context.get_minibatch_size()
  )));
  solve_threads.reserve(context.get_max_solve_threads());
  for(uint32 threads = 0; threads < context.get_max_solve_threads(); ++threads){
    solvers.push_back(make_unique<Solution_solver>(*net_solution, service_context, train_set.get_sequence_size()));
    process_threads[threads].reserve(context.get_max_processing_threads());
  }
  if(train_set.get_feature_size() != solvers.back()->get_output_size())
    throw std::runtime_error("Network output size doesn't match size of provided labels!");
  weight_updater = Updater_factory::build_weight_updater(net,weight_updater_,context);
}

void Sparse_net_approximizer::collect_approximates_from_random_direction(void){
  sdouble32 error_initial;
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

  if( /* If the net should be re-evaluated based on the iteration */
    (loops_unchecked >= context.get_insignificant_iteration_count())
    ||(loops_unchecked > (train_set.get_error_sum()/context.get_step_size()))
    ||(loops_unchecked > (test_set.get_error_sum()/context.get_step_size()))
  ){ 
    evaluate(); /* calculate the error value for the current network in the testing and training datasets */
    loops_unchecked = 0;
  }
  error_initial = train_set.get_error_sum();
  train_set.push_state();

  /* decide a random direction to approximate the network on */
  for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
    weight_steps[weight_index] = (
      context.get_step_size() 
      * ( (static_cast<sdouble32>(rand()%201) / double_literal(200.0)) - double_literal(1.0) )
      * std::min(double_literal(1.0),error_initial)
    );
    weight_epsilon += std::pow(weight_steps[weight_index], double_literal(2.0));
    net.set_weight_table(weight_index, (net.weight_table(weight_index) - weight_steps[weight_index]) );
  };
  weight_updater->update_solution_with_weights(*net_solution);
  weight_epsilon = std::sqrt(weight_epsilon) * double_literal(2.0);

  /* see the error values at the negative end of the current direction */
  evaluate(
    train_set,sequence_start_index,context.get_minibatch_size(),
    start_index_inside_sequence,context.get_memory_truncation()
  );
  error_negative_direction = train_set.get_error_sum();

  /* see the error values at the positive end of the current direction */
  for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index)
    net.set_weight_table(weight_index, (net.weight_table(weight_index) + (weight_steps[weight_index] * double_literal(2.0))) );
  weight_updater->update_solution_with_weights(*net_solution);
  evaluate(
    train_set,sequence_start_index,context.get_minibatch_size(),
    start_index_inside_sequence,context.get_memory_truncation()
  );
  error_positive_direction = train_set.get_error_sum();

  // for(uint32 weight_index = 0; weight_index < net.weight_table_size(); ++weight_index){
  //   std::cout << "weight index:" << weight_index
  //   << "\n errors: (+)" << error_positive_direction << "<>(-)" << error_negative_direction
  //   << "; step_size: " << ( context.get_step_size() )
  //   << "; weight delta: " << ( weight_steps[weight_index] )
  //   << "= (" << context.get_step_size() << "*" << std::min(double_literal(1.0),error_initial) << "* rand()"
  //   << ")"
  //   << "; error delta: " << ( error_positive_direction - error_negative_direction )
  //   << "; current_epsilon_double: " << weight_epsilon
  //   << "; sequences_to_evaluate: " << train_set.get_number_of_sequences()
  //   << "; gradient: " << ( error_positive_direction - error_negative_direction ) / (weight_epsilon)
  //   << std::endl << "=====" << std::endl;
  // }

  /* collect the fragment, revert weight changes */
  for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
    weight_gradients[weight_index] = (
      ( error_positive_direction - error_negative_direction ) / (weight_epsilon)
    );
    add_to_fragment( weight_index, (weight_gradients[weight_index] * weight_steps[weight_index]) );
    net.set_weight_table(weight_index, (net.weight_table(weight_index) - (weight_steps[weight_index])) );
  }
  // if(
  //   (error_initial > error_positive_direction)
  //   ||(error_initial > error_negative_direction)
  // ){
  //   for(uint32 weight_index = 0; weight_index < net.weight_table_size(); ++weight_index){
  //     weight_gradients[weight_index] = (
  //       ( error_positive_direction - error_negative_direction ) / (weight_epsilon)
  //     );
  //     add_to_fragment( weight_index, (weight_gradients[weight_index] * weight_steps[weight_index]) );
  //     net.set_weight_table(weight_index, (net.weight_table(weight_index) - (weight_steps[weight_index])) );
  //   }
  // }else{ /* Only revert the changes because current direction did not improve */
  //   for(uint32 weight_index = 0; weight_index < net.weight_table_size(); ++weight_index){
  //     net.set_weight_table(weight_index, (net.weight_table(weight_index) - (weight_steps[weight_index])) );
  //   }
  // }
  weight_updater->update_solution_with_weights(*net_solution);
  train_set.pop_state();
  ++loops_unchecked;
}

void Sparse_net_approximizer::collect_fragment(void){

  if(
    (loops_unchecked >= context.get_insignificant_iteration_count())
    ||(loops_unchecked > (train_set.get_error_sum()/context.get_step_size()))
    ||(loops_unchecked > (test_set.get_error_sum()/context.get_step_size()))
  ){
    /* calculate the error value for the current network in the testing and training datasets */
    evaluate();
    loops_unchecked = 0;
  }

  /* Approximate the gradient for every weight */
  vector<sdouble32> weight_gradients(net.weight_table_size(),double_literal(0.0));
  uint32 index_of_biggest = double_literal(0.0);
  sdouble32 average_gradient = double_literal(0.0);
  sdouble32 sum_gradient = double_literal(0.0);
  for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
    weight_gradients[weight_index] = get_gradient_fragment(weight_index);
    average_gradient += weight_gradients[weight_index];
    sum_gradient += weight_gradients[weight_index];
    if(weight_gradients[index_of_biggest] < weight_gradients[weight_index])
      index_of_biggest = weight_index;
  }
  average_gradient /= static_cast<sdouble32>(net.weight_table_size());
  add_to_fragment(index_of_biggest, (weight_gradients[index_of_biggest] / sum_gradient) );
  ++loops_unchecked;
}

sdouble32 Sparse_net_approximizer::get_gradient_fragment(uint32 weight_index){
  sdouble32 gradient;
  const sdouble32 current_epsilon = (
    std::sqrt(context.get_epsilon()) * (train_set.get_error_sum()/net.weight_table_size()  )
  );
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
  evaluate(
    train_set,sequence_start_index,context.get_minibatch_size(),
    start_index_inside_sequence,context.get_memory_truncation()
  );
  gradient = train_set.get_error_sum();

  /* Push it in other direction */
  net.set_weight_table(weight_index, (net.weight_table(weight_index) - current_epsilon_double) );
  weight_updater->update_solution_with_weights(*net_solution);

  /* Approximate the newly modified weights gradient */
  evaluate(
    train_set,sequence_start_index,context.get_minibatch_size(),
    start_index_inside_sequence,context.get_memory_truncation()
  );

  gradient = -(gradient - train_set.get_error_sum()) / (current_epsilon_double * train_set.get_number_of_sequences());

  /* Revert weight modification and the error state with it */
  net.set_weight_table(weight_index, (net.weight_table(weight_index) + current_epsilon) );
  weight_updater->update_solution_with_weights(*net_solution);
  train_set.pop_state();

  return gradient;
}

void Sparse_net_approximizer::evaluate(
  Data_aggregate& data_set, uint32 label_start_index, uint32 labels_to_eval,
  uint32 start_index_in_sequence, uint32 sequence_truncation
){
  uint32 sequence_index = label_start_index;
  const uint32 sequences_to_evaluate = std::min(data_set.get_number_of_sequences(), labels_to_eval);
  const uint32 sequences_in_one_thread = 1 + static_cast<uint32>(sequences_to_evaluate/context.get_max_solve_threads());

  for(uint32 thread_index = 0; ((thread_index < context.get_max_solve_threads())&&(data_set.get_number_of_sequences() > sequence_index)); ++thread_index){
    solve_threads.push_back(thread( /* As long as there are threads to open for remaining weights, open threads */
      &Sparse_net_approximizer::evaluate_thread, this, ref(data_set), thread_index, sequence_index,
      std::min(sequences_in_one_thread, (data_set.get_number_of_sequences() - sequence_index)),
      start_index_in_sequence, sequence_truncation
    ));
    sequence_index += sequences_in_one_thread;
  }
  wait_for_threads(solve_threads);
}

void Sparse_net_approximizer::evaluate_thread(
  Data_aggregate& data_set, uint32 solve_thread_index,
  uint32 sequence_start_index, uint32 sequences_to_evaluate,
  uint32 start_index_in_sequence, uint32 sequence_truncation
){
  uint32 raw_inputs_index;
  uint32 raw_label_index;

  for(uint32 sample = 0; sample < sequences_to_evaluate; ++sample){
    raw_label_index = sequence_start_index + sample;
    raw_inputs_index = raw_label_index * (data_set.get_sequence_size() + data_set.get_prefill_inputs_number());
    raw_label_index = raw_label_index * data_set.get_sequence_size();

    /* Prefill network with the initial inputs */
    solvers[solve_thread_index]->reset();
    for(uint32 prefill_iterator = 0; prefill_iterator < data_set.get_prefill_inputs_number(); ++prefill_iterator){
      solvers[solve_thread_index]->solve(data_set.get_input_sample(raw_inputs_index));
      ++raw_inputs_index;
    }

    /* Evaluate the current sequence step by step */
    for(uint32 sequence_iterator = 0; sequence_iterator < data_set.get_sequence_size(); ++sequence_iterator){
      solvers[solve_thread_index]->solve(data_set.get_input_sample(raw_inputs_index)); /* Solve the network for the sampled labels input */
      ++raw_label_index;
      ++raw_inputs_index;
    }
    lock_guard<mutex> my_lock(dataset_mutex);
    data_set.set_features_for_labels(
      solvers[solve_thread_index]->get_neuron_memory().get_whole_buffer(), start_index_in_sequence,
      ((sequence_start_index + sample) * data_set.get_sequence_size()) + start_index_in_sequence,
      sequence_truncation /* To avoid vanishing gradients, error calculation is truncated */
    ); /* Re-calculate error for the training set */
  }
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
      gradient_fragment.mutable_weight_synapses(weight_synapse_index_target)->set_interval_size(
        synapse_size + 1
      );
      insert_element_at_position(
        *gradient_fragment.mutable_values(), gradient_fragment_value, (values_index_target + synapse_size)
      );
    }
  }
}

void Sparse_net_approximizer::apply_fragment(void){
  uint32 fragment_value_index = 0;
  Synapse_iterator<>::iterate(gradient_fragment.weight_synapses(), [&](
    Index_synapse_interval interval, sint32 weight_index
  ){  
    net.set_weight_table(
      weight_index, ( net.weight_table(weight_index) - (gradient_fragment.values(fragment_value_index) * context.get_step_size()) )
    );
    ++fragment_value_index;
  });
  gradient_fragment = Gradient_fragment();
  loops_unchecked = context.get_insignificant_iteration_count() + 1;
}

} /* namespace sparse_net_library */
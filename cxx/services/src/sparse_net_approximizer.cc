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

#include "services/sparse_net_approximizer.h"

#include "gen/common.pb.h"
#include "services/synapse_iterator.h"

namespace sparse_net_library{

void Sparse_net_approximizer::collect(void){
  uint32 sequence_index;
  const uint32 sequences_in_one_thread = 1 + static_cast<uint32>(net.weight_table_size()/context.get_max_solve_threads());

  /* Collect the error value for the current network */
  for( /* As long as there are threads to open or remaining weights */
    uint32 thread_index = 0; 
    ( (thread_index < context.get_max_solve_threads())
      &&(static_cast<uint32>(train_set.get_number_of_sequences()) > sequence_index) );
    ++thread_index
  ){
    solve_threads.push_back(thread(
      &Sparse_net_approximizer::collect_thread, this, thread_index, sequence_index,
      std::min(sequences_in_one_thread, (net.weight_table_size() - sequence_index))
    ));
    sequence_index += sequences_in_one_thread;
  }
  wait_for_threads(solve_threads);
  initial_error = train_set.get_error();

  /* Modify a random weight */
  uint32 weight_index = rand()%(net.weight_table_size());
  net.set_weight_table(
    weight_index, (net.weight_table(weight_index) + context.get_step_size() / 2.0)
  );
  weight_updater->update_solution_with_weights(*net_solution);

  /* Approximate its gradient */
  for( /* As long as there are threads to open or remaining weights */
    uint32 thread_index = 0; 
    ( (thread_index < context.get_max_solve_threads())
      &&(static_cast<uint32>(train_set.get_number_of_sequences()) > sequence_index) );
    ++thread_index
  ){
    solve_threads.push_back(thread(
      &Sparse_net_approximizer::collect_thread, this, thread_index, sequence_index,
      std::min(sequences_in_one_thread, (net.weight_table_size() - sequence_index))
    ));
    sequence_index += sequences_in_one_thread;
  }
  wait_for_threads(solve_threads);
  
  add_to_fragment( /* Add the collected gradient into the fragment */
    weight_index, ((initial_error - train_set.get_error()) * context.get_step_size())
  );

  /* Revert weight modification */
  net.set_weight_table(
    weight_index, (net.weight_table(weight_index) - context.get_step_size() / 2.0)
  );
  weight_updater->update_solution_with_weights(*net_solution);

}

void Sparse_net_approximizer::collect_thread(uint32 solve_thread_index, uint32 sequence_index, uint32 sequences_to_evaluate){
  uint32 sample_index;

  if(train_set.get_feature_size() != solvers[solve_thread_index]->get_output_size())
    throw std::runtime_error("Network output size doesn't match size of provided labels!");

  for(uint32 sample = 0; sample < sequences_to_evaluate; ++sample){
    sample_index = (rand()%(train_set.get_number_of_sequences())) * train_set.get_sequence_size();

    /* Evaluate the current sequence step by step */
    solvers[solve_thread_index]->reset();
    for(uint32 sequence_iterator = 0; sequence_iterator < train_set.get_sequence_size(); ++sequence_iterator){
      solvers[solve_thread_index]->solve(train_set.get_input_sample(sample_index)); /* Solve the network for the sampled labels input */
      train_set.set_feature_for_label(sample_index, solvers[0]->get_neuron_data(0)); /* Re-calculate error for the training set */
      ++sample_index;
    }
    solvers[solve_thread_index]->reset();
  }
}

void Sparse_net_approximizer::add_to_fragment(uint32 weight_index, sdouble32 gradient_fragment_value){
  uint32 values_index = 0;
  uint32 weight_synapse_index = gradient_fragment.weight_synapses_size();
  for(weight_synapse_index = 0; static_cast<sint32>(weight_synapse_index) < gradient_fragment.weight_synapses_size(); ++weight_synapse_index){
    if( /* If the weight synapse is at or in-between the first index before the start of the synapse */
      ((gradient_fragment.weight_synapses(weight_synapse_index).starts()-1) >= static_cast<sint32>(weight_index))
      &&( /* and the one after the last index */
        (gradient_fragment.weight_synapses(weight_synapse_index).starts() + gradient_fragment.weight_synapses(weight_synapse_index).interval_size())
        <= weight_index
      )
    ){ /* current weight index shall be inside the fragment */
      gradient_fragment.mutable_weight_synapses(weight_synapse_index)->set_interval_size(
        gradient_fragment.weight_synapses(weight_synapse_index).interval_size() + 1
      );
      if((gradient_fragment.weight_synapses(weight_synapse_index).starts()-1) == static_cast<sint32>(weight_index)){
        /* The index is the first index before the synapse */
        insert_element_at_position(*gradient_fragment.mutable_values(),gradient_fragment_value,values_index);
      }else if(
        (gradient_fragment.weight_synapses(weight_synapse_index).starts() + gradient_fragment.weight_synapses(weight_synapse_index).interval_size())
        > weight_index
      ){ /* the index is inside the synapse */
        insert_element_at_position(
          *gradient_fragment.mutable_values(), gradient_fragment_value,
          (values_index + (
            (gradient_fragment.weight_synapses(weight_synapse_index).starts() + gradient_fragment.weight_synapses(weight_synapse_index).interval_size())
            - weight_index
          ))
        );
      }else{ /* The index is the first index after the synapse */
        insert_element_at_position(
          *gradient_fragment.mutable_values(), gradient_fragment_value,
          (values_index + gradient_fragment.weight_synapses(weight_synapse_index).interval_size())
        );
      }
      break; /* weight is placed inside the fragment, no need to continue */
    }
    values_index += gradient_fragment.weight_synapses(weight_synapse_index).interval_size();
  }
  if(
    (0 == gradient_fragment.weight_synapses_size())
    ||(static_cast<sint32>(weight_synapse_index) < gradient_fragment.weight_synapses_size())
  ){
    gradient_fragment.add_values(gradient_fragment_value);
    tmp_synapse_interval.set_interval_size(1);
    tmp_synapse_interval.set_starts(weight_index);
    *gradient_fragment.add_weight_synapses() = tmp_synapse_interval;
  }
}

void Sparse_net_approximizer::apply_fragment(void){
  uint32 fragment_value_index = 0;
  Synapse_iterator<>::iterate(gradient_fragment.weight_synapses(), [&](
    Index_synapse_interval interval, sint32 weight_index
  ){
    net.set_weight_table(
      weight_index,
      (net.weight_table(weight_index) - gradient_fragment.values(fragment_value_index) * context.get_step_size())
    );
    ++fragment_value_index;
  });

  gradient_fragment = Gradient_fragment();
}

} /* namespace sparse_net_library */
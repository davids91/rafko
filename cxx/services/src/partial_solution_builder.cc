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

#include "services/partial_solution_builder.h"

namespace sparse_net_library{

void Partial_solution_builder::add_neuron_to_partial_solution(uint32 neuron_index){
  if(net.neuron_array_size() > static_cast<int>(neuron_index)){
    const Neuron& neuron = net.neuron_array(neuron_index);
    Synapse_interval temp_synapse_interval;
    Synapse_iterator<> weight_iterator(neuron.input_weights());
    Synapse_iterator<> index_iterator(neuron.input_indices());
    /* Add a new Neuron into the partial solution */
    partial.set_internal_neuron_number(partial.internal_neuron_number() + 1);

    /* Copy in Neuron parameters */
    partial.add_neuron_transfer_functions(neuron.transfer_function_idx());
    partial.add_memory_filter_index(partial.weight_table_size());
    partial.add_weight_table(net.weight_table(neuron.memory_filter_idx()));

    /* Copy in weights from the net */
    partial.add_weight_synapse_number(neuron.input_weights_size());
    weight_iterator.iterate([&](uint32 weight_synapse_size){
      temp_synapse_interval.set_starts(partial.weight_table_size());
      temp_synapse_interval.set_interval_size(weight_synapse_size);
      *partial.add_weight_indices() = temp_synapse_interval;
    },[&](sint32 weight_index){
      partial.add_weight_table(net.weight_table(weight_index));
    });

    /* Copy in input data references */
    neuron_synapse_count = 0;
    previous_neuron_input_source = neuron_input_none;
    previous_neuron_input_index = input_synapse.size(); /* Input value to point above the size of the input */
    uint32 index_synapse_previous_size = partial.inside_indices_size();
    index_iterator.iterate([&](sint32 neuron_input_index){ /* Put each Neuron input into the @Partial_solution */
      if(!look_for_neuron_input(neuron_input_index)){
        /* Check if the partial input synapse needs to be closed */
        if(!look_for_neuron_input_internally(neuron_input_index)){ /* if the Neuron is not found internally */
          if(
            (0 < partial_input_synapse_count)
            &&((
                Synapse_iterator<>::is_index_input(neuron_input_index)
                &&(input_synapse.back() != neuron_input_index+1)
              )||(
                input_synapse.back() != neuron_input_index-1
            ))
          ){
            partial_input_synapse_count = 0; /* Close synapse! */
          }
          if(0 < neuron_synapse_count){
            if(
              (neuron_input_external != previous_neuron_input_source)
              ||(static_cast<int>(input_synapse.size()-1) != previous_neuron_input_index)
            )neuron_synapse_count = 0; /* Close synapse! */
          }
          previous_neuron_input_index = input_synapse.size(); /* Update previous neuron input source as well */
          previous_neuron_input_source = neuron_input_external;/* since the input was added to be taken from the @Partial_solution inputs */
          add_to_synapse( /* Neural input shall be added from the input of the @Partial_solution */
            Synapse_iterator<>::synapse_index_from_input_index(input_synapse.size()),
            neuron_synapse_count, partial.mutable_inside_indices()
          );
          add_to_synapse(
            neuron_input_index, partial_input_synapse_count,
            partial.mutable_input_data()
          );
        }/* Neuron input was found internally in the @Partial_solution */
      }/* Neuron input was found in the @Partial_solution inputs, continue to look for it.. */
    });

    if(0 < (partial.inside_indices_size() - index_synapse_previous_size))
      partial.add_index_synapse_number(partial.inside_indices_size() - index_synapse_previous_size);

    if( /* In case th latest input synapse is of 0 length, remove it */
      (0 < partial.input_data_size())
      &&(0 == partial.input_data(partial.input_data_size()-1).interval_size())
    )partial.mutable_input_data()->RemoveLast();

  }else throw "Neuron index is out of bounds from net neuron array!";
}

bool Partial_solution_builder::look_for_neuron_input(int neuron_input_index){
  uint32 candidate_synapse_index = input_synapse.size();

  input_synapse.iterate_terminatable([&](int synapse_index){
    if(candidate_synapse_index == input_synapse.size()) candidate_synapse_index = 0;
    if(synapse_index == neuron_input_index){
      return false; /* No need to continue Synapse iteration, found the right candidate! */
    }else{
      ++candidate_synapse_index; /* Step the candidate iterator forward to the next index in the input array */
      return true;
    }
  });
  if(candidate_synapse_index < input_synapse.size()){ /* Found the neuron input in the candidate synapse inputs */
    /* Check if the newly added Neuron synapse can be continued based on value, or a new Synapse needs to be added */
    if(0 < neuron_synapse_count){
      if(
        (neuron_input_external != previous_neuron_input_source)
        ||(static_cast<int>(candidate_synapse_index-1) != previous_neuron_input_index)
      )neuron_synapse_count = 0; /* Close synapse! */
    }
    previous_neuron_input_index = candidate_synapse_index;
    previous_neuron_input_source = neuron_input_external;
    add_to_synapse(
      Synapse_iterator<>::synapse_index_from_input_index(candidate_synapse_index), neuron_synapse_count,
      partial.mutable_inside_indices()
    );
    return true;
  }else return false; /* couldn't find the Neuron input in the @Partial solution input synapses */
}

bool Partial_solution_builder::look_for_neuron_input_internally(uint32 neuron_input_index){

  using std::for_each;
  
  Synapse_iterator<> partial_neuron_output_index(partial.output_data());

  if(0 < partial_neuron_output_index.size()){
    uint32 inner_neuron_index = 0;
    for(uint32 internal_neuron_iterator = 0; internal_neuron_iterator < partial.internal_neuron_number(); ++internal_neuron_iterator){
      if(static_cast<sint32>(neuron_input_index) != partial_neuron_output_index[internal_neuron_iterator])++inner_neuron_index;
      else{
        if(0 < neuron_synapse_count){
          if(
            (neuron_input_internal != previous_neuron_input_source)
            ||(static_cast<int>(inner_neuron_index)-1 != previous_neuron_input_index)
          ){
            neuron_synapse_count = 0; /* Close synapse! */
          }
        }
        previous_neuron_input_index = inner_neuron_index;
        previous_neuron_input_source = neuron_input_internal;
        add_to_synapse( /* The Neuron input points to an internal Neuron (no conversion to input synapse index) */
          inner_neuron_index, neuron_synapse_count,
          partial.mutable_inside_indices()
        );
        return true;
      }
    }

  }
  return false;
}

} /* namespace sparse_net_library */

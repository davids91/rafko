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

#include "rafko_net/services/partial_solution_builder.h"

#include <stdexcept>

#include "rafko_net/models/input_function.h"

namespace rafko_net{

uint32 PartialSolutionBuilder::neuron_synapse_count = 0;
uint32 PartialSolutionBuilder::partial_input_synapse_count = 0;
sint32 PartialSolutionBuilder::previous_neuron_input_index;
uint8 PartialSolutionBuilder::previous_neuron_input_source;

std::pair<uint32,uint32> PartialSolutionBuilder::add_neuron_to_partial_solution(const RafkoNet& net, uint32 neuron_index, PartialSolution& partial){
  SynapseIterator<InputSynapseInterval> input_synapse(partial.input_data());
  previous_neuron_input_index = input_synapse.size();
  previous_neuron_input_source = neuron_input_none;
  partial_input_synapse_count = 0;
  neuron_synapse_count = 0;

  if(net.neuron_array_size() > static_cast<int>(neuron_index)){
    uint32 max_reach_back = 0;
    uint32 max_reach_index = 0;
    IndexSynapseInterval temp_synapse_interval;
    SynapseIterator<> weight_iterator(net.neuron_array(neuron_index).input_weights());
    SynapseIterator<InputSynapseInterval> input_iterator(net.neuron_array(neuron_index).input_indices());
    partial.mutable_output_data()->set_interval_size(partial.output_data().interval_size() + 1u);

    /* Copy in Neuron parameters and weights from the net */
    partial.add_neuron_input_functions(net.neuron_array(neuron_index).input_function());
    partial.add_neuron_transfer_functions(net.neuron_array(neuron_index).transfer_function());
    partial.add_weight_synapse_number(net.neuron_array(neuron_index).input_weights_size());
    weight_iterator.iterate([&](IndexSynapseInterval weight_synapse){
      temp_synapse_interval.set_starts(partial.weight_table_size());
      temp_synapse_interval.set_interval_size(weight_synapse.interval_size());
      *partial.add_weight_indices() = temp_synapse_interval;
    },[&](sint32 weight_index){
      partial.add_weight_table(net.weight_table(weight_index));
    });

    /* Copy in input data references */
    neuron_synapse_count = 0;
    previous_neuron_input_source = neuron_input_none;
    previous_neuron_input_index = input_synapse.size(); /* Input value to point above the size of the input */
    const uint32 index_synapse_previous_size = partial.inside_indices_size();

    uint32 current_backreach;
    input_iterator.iterate([&](InputSynapseInterval interval_synapse){
      current_backreach = interval_synapse.reach_past_loops();
      if(interval_synapse.reach_past_loops() > max_reach_back)
        max_reach_back = interval_synapse.reach_past_loops();
      if(SynapseIterator<InputSynapseInterval>::is_synapse_input(interval_synapse)){
        uint32 input_index = SynapseIterator<InputSynapseInterval>::synapse_index_from_input_index(interval_synapse.starts()) + interval_synapse.interval_size() - 1u;
        if(max_reach_index < input_index)max_reach_index = input_index;
      }
    },[&](sint32 neuron_input_index){ /* Put each Neuron input into the @PartialSolution */
      if(!look_for_neuron_input(neuron_input_index, current_backreach, input_synapse, partial)){
        /* Check if the partial input synapse needs to be closed */
        if( /* if the Neuron has any inputs from the past or not found internally */
          (0 < current_backreach)||(!look_for_neuron_input_internally(neuron_input_index, partial))
        ){
          if( /* Close input synapse if */
            (0 < partial_input_synapse_count) /* There is one open already */
            &&(( /* The latest index in the input synapse isn't the preceeding index of the current index */
                SynapseIterator<>::is_index_input(neuron_input_index)
                &&(input_synapse.back() != neuron_input_index+1)
              )||(
                (!SynapseIterator<>::is_index_input(neuron_input_index))
                &&(input_synapse.back() != neuron_input_index-1)
              )||(input_synapse.last_synapse().reach_past_loops() != current_backreach) /* Current index not in same memory depth */
            )
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
          previous_neuron_input_source = neuron_input_external;/* since the input was added to be taken from the @PartialSolution inputs */
          add_to_synapse( /* Neural input shall be added from the input of the @PartialSolution */
            SynapseIterator<>::synapse_index_from_input_index(input_synapse.size()), 0,
            neuron_synapse_count, partial.mutable_inside_indices()
          );
          add_to_synapse(neuron_input_index, current_backreach,partial_input_synapse_count, partial.mutable_input_data());
        }/* Neuron input was found internally in the @PartialSolution */
      }/* Neuron input was found in the @PartialSolution inputs, continue to look for it.. */
    });

    if(0 < (partial.inside_indices_size() - index_synapse_previous_size))
      partial.add_index_synapse_number(partial.inside_indices_size() - index_synapse_previous_size);

    if( /* In case th latest input synapse is of 0 length, remove it */
      (0 < partial.input_data_size())
      &&(0 == partial.input_data(partial.input_data_size()-1).interval_size())
    )partial.mutable_input_data()->RemoveLast();

    return std::make_pair(max_reach_back, max_reach_index);
  }else throw std::runtime_error("Neuron index is out of bounds from net neuron array!");
}

bool PartialSolutionBuilder::look_for_neuron_input(
  sint32 neuron_input_index, uint32 input_reach_back,
  SynapseIterator<InputSynapseInterval>& input_synapse, PartialSolution& partial
){
  uint32 candidate_synapse_index = input_synapse.size();
  uint32 current_backreach;
  input_synapse.iterate_terminatable([&](InputSynapseInterval interval_synapse){
    current_backreach = interval_synapse.reach_past_loops();
    return true;
  },[&](sint32 synapse_index){
    if(candidate_synapse_index == input_synapse.size()) candidate_synapse_index = 0;
    if((input_reach_back == current_backreach)&&(synapse_index == neuron_input_index)){ /* If the index as well as the time of input matches */
      return false; /* No need to continue Synapse iteration, found the right candidate! */
    }else{ /* Step the candidate iterator forward to the next index in the input array */
      ++candidate_synapse_index;
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
    add_to_synapse( /* inside indices always taking input from the current value */
      SynapseIterator<>::synapse_index_from_input_index(candidate_synapse_index),
      0, neuron_synapse_count, partial.mutable_inside_indices()
    );
    return true;
  }else return false; /* couldn't find the Neuron input in the @Partial solution input synapses */
}

bool PartialSolutionBuilder::look_for_neuron_input_internally(uint32 neuron_input_index, PartialSolution& partial){
  if(
    (static_cast<sint32>(neuron_input_index) >= partial.output_data().starts())
    &&(neuron_input_index < (partial.output_data().starts() + partial.output_data().interval_size()))
  ){
    const uint32 inner_neuron_index = (neuron_input_index - partial.output_data().starts());
    if( /* there is a synapse already open for the current Neuron input */
      (0 < neuron_synapse_count) /* ..and the current found index can not continue it */
      &&((neuron_input_internal != previous_neuron_input_source)||(static_cast<int>(inner_neuron_index)-1 != previous_neuron_input_index))
    )neuron_synapse_count = 0; /* Close synapse! */
    previous_neuron_input_index = inner_neuron_index;
    previous_neuron_input_source = neuron_input_internal;
    add_to_synapse( /* The Neuron input points to an internal Neuron (no conversion to input synapse index) */
      inner_neuron_index, 0, neuron_synapse_count, partial.mutable_inside_indices()
    );
    return true;
  }else return false;
}

} /* namespace rafko_net */

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

#include "sparse_net_library/services/partial_solution_solver.h"

#include <algorithm>
#include <cmath>

#include "sparse_net_library/models/transfer_function.h"
#include "sparse_net_library/models/spike_function.h"

namespace sparse_net_library {

void Partial_solution_solver::collect_input_data(const vector<sdouble32>& input_data){
  uint32 index = 0;
  input_iterator.iterate([&](Input_synapse_interval input_synapse, sint32 synapse_index){
    if(Synapse_iterator<>::is_index_input(synapse_index)){ /* If @Partial_solution input is from the network input */
      collected_input_data[index] = input_data[Synapse_iterator<>::input_index_from_synapse_index(synapse_index)];
    }else if(static_cast<sint32>(neuron_data.buffer_size()) > synapse_index){  /* If @Partial_solution input is from the previous row */
      collected_input_data[index] = neuron_data.get_element(synapse_index,input_synapse.reach_past_loops());
    }
    ++index;
  });
}

void Partial_solution_solver::solve(){
  sdouble32 new_neuron_data = 0;
  sdouble32 new_neuron_input;
  uint32 weight_synapse_iterator_start = 0; /* Which is the first synapse belonging to the neuron under @neuron_iterator */
  uint32 input_synapse_iterator_start = 0; /* Which is the first synapse belonging to the neuron under @neuron_iterator */
  uint32 input_synapse_index = 0; /* Which synapse is being processed inside the Neuron */
  uint32 input_index_offset = 0;
  for(uint16 neuron_iterator = 0; neuron_iterator < detail.internal_neuron_number(); ++neuron_iterator){
    new_neuron_data = 0;
    internal_iterator.iterate([&](Index_synapse_interval weight_synapse, sint32 weight_index){
      /* Collect input only as long as it's assigned to the current inner neuron */
      if(detail.index_synapse_number(neuron_iterator) > input_synapse_index){
        /* Neuron gets its input from the partial solution input */
        if(Synapse_iterator<>::is_index_input(detail.inside_indices(input_synapse_iterator_start + input_synapse_index).starts()))
          new_neuron_input = collected_input_data[Synapse_iterator<>::input_index_from_synapse_index(
            detail.inside_indices(input_synapse_iterator_start + input_synapse_index).starts() - input_index_offset
          )];
        else new_neuron_input = neuron_data.get_element(0)[ /* Neuron gets its input internaly */
          detail.output_data().starts() + detail.inside_indices(input_synapse_iterator_start + input_synapse_index).starts() + input_index_offset
        ]; 
        ++input_index_offset; /* Step the input index to the next input */
        if(input_index_offset >= detail.inside_indices(input_synapse_iterator_start + input_synapse_index).interval_size()){
          input_index_offset = 0; /* In case the next input would ascend above the current patition, go to next one */
          ++input_synapse_index;
          /*!Note: It is possible to increase the @weight_synapse_index above @detail.weight_synapse_number(neuron_iterator)
           * but that isn't chekced here, mainly for performance reasons.
           **/
        }
      }else /* Every input above the number assigned to the internal Neuron shall count as an input of 1.0 */
        new_neuron_input = double_literal(1.0);

      /* The weighted input shall be added to the calculated value */
      new_neuron_data += new_neuron_input * detail.weight_table(weight_index);
    },weight_synapse_iterator_start, detail.weight_synapse_number(neuron_iterator));
    weight_synapse_iterator_start += detail.weight_synapse_number(neuron_iterator);
    input_synapse_iterator_start += input_synapse_index;
    input_index_offset = 0;
    input_synapse_index = 0;

    /* Save transfer function input value and apply transfer function */
    transfer_function_input[neuron_iterator] = new_neuron_data;
    new_neuron_data = transfer_function.get_value(
      detail.neuron_transfer_functions(neuron_iterator), new_neuron_data
    );

    /* Save transfer funtion output value and apply spike function */
    transfer_function_output[neuron_iterator] = new_neuron_data;
    neuron_data.get_element(0)[detail.output_data().starts() + neuron_iterator] = Spike_function::get_value(
      detail.weight_table(detail.memory_filter_index(neuron_iterator)),
      new_neuron_data,
      neuron_data.get_const_element(0)[detail.output_data().starts() + neuron_iterator]
    );
  } /* Go through the neurons */
}

void Partial_solution_solver::provide_gradient_data(vector<sdouble32>& transfer_function_input_, vector<sdouble32>& transfer_function_output_) const{
  if(transfer_function_input.size() != transfer_function_output.size()) throw std::runtime_error("Neuron gradient data Incompatible!");
  uint32 output_index_start = 0;
    std::copy(
      transfer_function_input.begin() + output_index_start,
      transfer_function_input.begin() + output_index_start + detail.output_data().interval_size(),
      transfer_function_input_.begin() + detail.output_data().starts()
    );
    std::copy(
      transfer_function_output.begin() + output_index_start,
      transfer_function_output.begin() + output_index_start + detail.output_data().interval_size(),
      transfer_function_output_.begin() + detail.output_data().starts()
    );
}

bool Partial_solution_solver::is_valid(void) const{
  if(
    (0u < detail.internal_neuron_number())
    &&(static_cast<int>(detail.internal_neuron_number()) == detail.index_synapse_number_size())
    &&(static_cast<int>(detail.internal_neuron_number()) == detail.weight_synapse_number_size())
    &&(static_cast<int>(detail.internal_neuron_number()) == detail.neuron_transfer_functions_size())
    &&(static_cast<int>(detail.internal_neuron_number()) == detail.memory_filter_index_size())
  ){
    int weight_synapse_number = 0;
    int index_synapse_number = 0;

    for(uint16 neuron_iterator = 0u; neuron_iterator < detail.internal_neuron_number(); ++neuron_iterator){
      weight_synapse_number += detail.weight_synapse_number(neuron_iterator); /* Calculate how many inputs the neuron shall have altogether */
      index_synapse_number += detail.index_synapse_number(neuron_iterator); /* Calculate how many inputs the neuron shall have altogether */
    }

    if((0 < index_synapse_number)&&(0 < weight_synapse_number)){
      /* Check if the inputs for every Neuron are before its index.
       * This will ensure that there are no unresolved dependencies are present at any Neuron
       **/
      uint32 index_synapse_iterator_start = 0;
      uint32 count_of_input_indexes = 0;
      uint32 weight_synapse_iterator_start = 0;
      uint32 count_of_input_weights = 0;
      for(uint32 neuron_iterator = 0; neuron_iterator < detail.internal_neuron_number(); neuron_iterator++){
        count_of_input_indexes = 0;
        count_of_input_weights = 0;
        for(uint32 neuron_input_iterator = 0; neuron_input_iterator < detail.index_synapse_number(neuron_iterator); ++neuron_input_iterator){
          count_of_input_indexes += detail.inside_indices(index_synapse_iterator_start + neuron_input_iterator).interval_size();
          if( /* If a synapse input in a Neuron points after the neurons index */
            (detail.inside_indices(index_synapse_iterator_start + neuron_input_iterator).starts()
             + detail.inside_indices(index_synapse_iterator_start + neuron_input_iterator).interval_size() ) >= neuron_iterator
          ){ /* Self-recurrence is simulated by adding the current data of a neuron as an input into the solution detail */
            return false;
          }

          /* Check if the number of weights match the number of input indexes for every Neuron */
          for(uint32 neuron_weight_iterator = 0; neuron_weight_iterator < detail.weight_synapse_number(neuron_iterator); ++neuron_weight_iterator){
            count_of_input_weights +=  detail.weight_indices(weight_synapse_iterator_start + neuron_weight_iterator).interval_size();
          }
          weight_synapse_iterator_start += detail.weight_synapse_number(neuron_iterator);
          index_synapse_iterator_start += detail.index_synapse_number(neuron_iterator);

          if(count_of_input_indexes == count_of_input_weights){
            return false;
          }
        }
      }
    }else return false;

    return(
      (index_synapse_number == detail.inside_indices_size())
      &&(weight_synapse_number == detail.weight_indices_size())
    );
  }else return false;
}

} /* namespace sparse_net_library */

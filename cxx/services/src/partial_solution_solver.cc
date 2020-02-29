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

#include "services/partial_solution_solver.h"

#include <algorithm>
#include <cmath>

#include "models/transfer_function.h"
#include "models/spike_function.h"

namespace sparse_net_library {

void Partial_solution_solver::collect_input_data(vector<sdouble32>& input_data, vector<sdouble32>& neuron_data){
  uint32 index = 0;
  input_iterator.iterate([&](int synapse_index){
    if(Synapse_iterator::is_index_input(synapse_index)){ /* If @Partial_solution input is from the network input */
      collected_input_data[index] = input_data[Synapse_iterator::input_index_from_synapse_index(synapse_index)];
    }else if(neuron_data.size() > static_cast<std::size_t>(synapse_index)){  /* If @Partial_solution input is from the previous row */
      collected_input_data[index] = neuron_data[synapse_index];
    }
    ++index;
  });
}

void Partial_solution_solver::solve(){
  sdouble32 new_neuron_data = 0;
  sdouble32 new_neuron_input;
  uint32 index_synapse_iterator_start = 0; /* Which is the first synapse belonging to the neuron under @neuron_iterator */
  uint32 weight_synapse_index = 0; /* Which synapse is being processed inside the Neuron */
  uint32 weight_index = 0;

  for(uint16 neuron_iterator = 0; neuron_iterator < detail.internal_neuron_number(); ++neuron_iterator){
    new_neuron_data = 0;
    internal_iterator.iterate_unsafe([&](int synapse_index){
      if(Synapse_iterator::is_index_input(synapse_index)){ /* Neuron gets its input from the partialsolution input */
        new_neuron_input = collected_input_data[Synapse_iterator::input_index_from_synapse_index(synapse_index)];
      }else{ /* Neuron gets its input internaly */
        new_neuron_input = neuron_output[synapse_index];
      }

      new_neuron_data += new_neuron_input * /* Data of the input * weight of the input * */
      detail.weight_table(
        detail.weight_indices(weight_synapse_index).starts() + weight_index
      );

      ++weight_index; /* Step the Weight index forwards */
      if(weight_index >= detail.weight_indices(weight_synapse_index).interval_size()){
        weight_index = 0; /* In case the next weight would ascend above the current patition, go to next one */
        ++weight_synapse_index;

        /*!Note: It is possible, in case of an incorrect configuration that the indexes and synapses
         * don't match. It is possible to increase the @weight_synapse_index above @detail.weight_synapse_number(neuron_iterator)
         * but that isn't chekced here, mainly for performance reasons.
         **/
      }
    },index_synapse_iterator_start, detail.index_synapse_number(neuron_iterator));
    index_synapse_iterator_start += detail.index_synapse_number(neuron_iterator);

    /* Add bias */
    new_neuron_data += detail.weight_table(detail.bias_index(neuron_iterator));
      
    transfer_function_input[neuron_iterator] = new_neuron_data;

    /* Apply transfer function */
    new_neuron_data = transfer_function.get_value(
      detail.neuron_transfer_functions(neuron_iterator), new_neuron_data
    );

    transfer_function_output[neuron_iterator] = new_neuron_data;

    /* Apply spike function */
    neuron_output[neuron_iterator] = Spike_function::get_value(
      detail.weight_table(detail.memory_filter_index(neuron_iterator)),
      new_neuron_data,
      neuron_output[neuron_iterator]
    );
  } /* Go through the neurons */
}

uint32 Partial_solution_solver::get_input_size(void) const{
  return collected_input_data.size();
}

bool Partial_solution_solver::is_valid(void) const{
  if(
    (0u < detail.internal_neuron_number())
    &&(static_cast<int>(detail.internal_neuron_number()) == detail.index_synapse_number_size())
    &&(static_cast<int>(detail.internal_neuron_number()) == detail.weight_synapse_number_size())
    &&(static_cast<int>(detail.internal_neuron_number()) == detail.actual_index_size())
    &&(static_cast<int>(detail.internal_neuron_number()) == detail.neuron_transfer_functions_size())
    &&(static_cast<int>(detail.internal_neuron_number()) == detail.memory_filter_index_size())
    &&(static_cast<int>(detail.internal_neuron_number()) == detail.bias_index_size())
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
        for(uint32 internal_iterator = 0; internal_iterator < detail.index_synapse_number(neuron_iterator); ++internal_iterator){
          count_of_input_indexes += detail.inside_indices(index_synapse_iterator_start + internal_iterator).interval_size();
          if( /* If a synapse input in a Neuron points after the neurons index */
            (detail.inside_indices(index_synapse_iterator_start + internal_iterator).starts()
             + detail.inside_indices(index_synapse_iterator_start + internal_iterator).interval_size() ) >= neuron_iterator
          ){ /* Self-recurrence is simulated by adding the current data of a neuron as an input into the solution detail */
            return false;
          }

          /* Check if the number of weights match the number of input indexes for every Neuron */
          for(uint32 internal_iterator = 0; internal_iterator < detail.weight_synapse_number(neuron_iterator); ++internal_iterator){
            count_of_input_weights +=  detail.weight_indices(weight_synapse_iterator_start + internal_iterator).interval_size();
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

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
#include <stdexcept>
#include <cmath>

#include "sparse_net_library/models/transfer_function.h"
#include "sparse_net_library/models/spike_function.h"

namespace sparse_net_library {

DataPool<sdouble32> Partial_solution_solver::common_data_pool;

void Partial_solution_solver::solve(const vector<sdouble32>& input_data, DataRingbuffer& output_neuron_data,  vector<sdouble32>& temp_data) const{
  sdouble32 new_neuron_data = 0;
  sdouble32 new_neuron_input;
  uint32 weight_synapse_iterator_start = 0; /* Which is the first synapse belonging to the neuron under @neuron_iterator */
  uint32 input_synapse_iterator_start = 0; /* Which is the first synapse belonging to the neuron under @neuron_iterator */
  uint32 input_synapse_index = 0; /* Which synapse is being processed inside the Neuron */
  uint32 input_index_offset = 0;
  sint32 input_index;

  /* Collect the input data to solve the partial solution */
  input_iterator.skim([&](Input_synapse_interval input_synapse){
    if(Synapse_iterator<>::is_index_input(input_synapse.starts())){ /* If @Partial_solution input is from the network input */
      std::copy(
        input_data.begin() + Synapse_iterator<>::input_index_from_synapse_index(input_synapse.starts()),
        input_data.begin() + Synapse_iterator<>::input_index_from_synapse_index(input_synapse.starts()) + input_synapse.interval_size(),
        temp_data.begin() + input_index_offset
      );
    }else if(static_cast<sint32>(output_neuron_data.buffer_size()) > input_synapse.starts()){  /* If @Partial_solution input is from the previous row */
      std::copy(
        output_neuron_data.get_element(input_synapse.reach_past_loops()).begin() + input_synapse.starts(),
        output_neuron_data.get_element(input_synapse.reach_past_loops()).begin() + input_synapse.starts() + input_synapse.interval_size(),
        temp_data.begin() + input_index_offset
      );
    }
    input_index_offset += input_synapse.interval_size();
  });

  /* Solve the Partial Solution based on the collected input data and stored operations */
  input_index_offset = 0;
  for(uint16 neuron_iterator = 0; neuron_iterator < detail.internal_neuron_number(); ++neuron_iterator){
    new_neuron_data = 0;
    internal_weight_iterator.iterate([&](Index_synapse_interval weight_synapse, sint32 weight_index){
      if(detail.index_synapse_number(neuron_iterator) > input_synapse_index){ /* Collect input only as long as there is any in the current inner neuron */
        input_index = detail.inside_indices(input_synapse_iterator_start + input_synapse_index).starts();
        if(Synapse_iterator<>::is_index_input(input_index)){ /* Neuron gets its input from the partial solution input */
          input_index = Synapse_iterator<>::input_index_from_synapse_index(input_index - input_index_offset);
          new_neuron_input = temp_data[input_index];
        }else{ /* Neuron gets its input internaly */
          input_index = detail.output_data().starts() + input_index + input_index_offset;
          new_neuron_input = output_neuron_data.get_element(0)[input_index];
        }
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

    new_neuron_data = transfer_function.get_value( /* apply transfer function */
      detail.neuron_transfer_functions(neuron_iterator), new_neuron_data
    );

    new_neuron_data = Spike_function::get_value( /* apply spike function */
      detail.weight_table(detail.memory_filter_index(neuron_iterator)),
      new_neuron_data, output_neuron_data.get_element(0, (detail.output_data().starts() + neuron_iterator))
    );

    output_neuron_data.set_element( /* Store the resulting Neuron value */
      0, detail.output_data().starts() + neuron_iterator, new_neuron_data
    );
  } /* Go through the neurons */
}

bool Partial_solution_solver::is_valid(void) const{
  if(
    (0u < detail.internal_neuron_number())
    &&(static_cast<int>(detail.internal_neuron_number()) == detail.index_synapse_number_size())
    &&(static_cast<int>(detail.internal_neuron_number()) == detail.weight_synapse_number_size())
    &&(static_cast<int>(detail.internal_neuron_number()) == detail.neuron_transfer_functions_size())
    &&(static_cast<int>(detail.internal_neuron_number()) == detail.memory_filter_index_size())
  ){
    uint32 weight_synapse_number = 0;
    uint32 index_synapse_number = 0;

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
        for(uint32 input_iterator = 0; input_iterator < detail.index_synapse_number(neuron_iterator); ++input_iterator){
          count_of_input_indexes += detail.inside_indices(index_synapse_iterator_start + input_iterator).interval_size();
          if( /* If a synapse input in a Neuron points after the neurons index */
            (detail.inside_indices(index_synapse_iterator_start + input_iterator).starts()
             + detail.inside_indices(index_synapse_iterator_start + input_iterator).interval_size() ) >= neuron_iterator
          ){ /* Self-recurrence is simulated by adding the current data of a neuron as an input into the solution detail */
            return false;
          }

          /* Check if the number of weights match the number of input indexes for every Neuron */
          for(uint32 input_iterator = 0; input_iterator < detail.weight_synapse_number(neuron_iterator); ++input_iterator){
            count_of_input_weights +=  detail.weight_indices(weight_synapse_iterator_start + input_iterator).interval_size();
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
      (static_cast<sint32>(index_synapse_number) == detail.inside_indices_size())
      &&(static_cast<sint32>(weight_synapse_number) == detail.weight_indices_size())
    );
  }else return false;
}

} /* namespace sparse_net_library */

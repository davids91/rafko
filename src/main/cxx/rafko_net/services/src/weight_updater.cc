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

#include "rafko_net/services/weight_updater.h"

#include "rafko_net/services/synapse_iterator.h"

namespace rafko_net{

using rafko_mainframe::ServiceContext;

void WeightUpdater::calculate_velocity_thread(const vector<sdouble32>& gradients, uint32 weight_index, uint32 weight_number){
  for(uint32 weight_iterator = 0; weight_iterator < weight_number; ++weight_iterator){
    current_velocity[weight_index + weight_iterator] = get_new_velocity(weight_index + weight_iterator, gradients);
  }
}

void WeightUpdater::update_weight_with_velocity(uint32 weight_index, uint32 weight_number){
  for(uint32 weight_iterator = 0; weight_iterator < weight_number; ++weight_iterator){
    net.set_weight_table(
      weight_index + weight_iterator, get_new_weight(weight_index + weight_iterator)
    );
  }
}

void WeightUpdater::calculate_velocity(const vector<sdouble32>& gradients){
  execution_threads.start_and_block([this, &gradients](uint32 thread_index){
    sint32 weight_index_start = weights_to_do_in_one_thread * thread_index;
    if(weight_index_start < net.weight_table_size()){
      uint32 weight_index = (weights_to_do_in_one_thread * thread_index);
      calculate_velocity_thread(
        gradients, weight_index, std::min(weights_to_do_in_one_thread, (net.weight_table_size() - weight_index))
      );
    }
  });
}

void WeightUpdater::update_weights_with_velocity(void){
  execution_threads.start_and_block([this](uint32 thread_index){
    sint32 weight_index_start = weights_to_do_in_one_thread * thread_index;
    if(weight_index_start < net.weight_table_size()){
      uint32 weight_index = (weights_to_do_in_one_thread * thread_index);
      update_weight_with_velocity(weight_index, std::min(weights_to_do_in_one_thread, (net.weight_table_size() - weight_index)));
    }
  });
}

void WeightUpdater::update_solution_with_weight(Solution& solution, uint32 weight_index) const{
  /* Iterate through the neurons in the network */
  for(uint32 neuron_index = 0; static_cast<sint32>(neuron_index) < net.neuron_array_size(); ++neuron_index){
    bool is_neuron_relevant_to_weight = false;
    /* iterate through the weights of the current neuron */
    SynapseIterator<>::skim_terminatable(net.neuron_array(neuron_index).input_weights(),
    [&is_neuron_relevant_to_weight, weight_index](IndexSynapseInterval input_weight_synapse){
      if(
        (static_cast<sint32>(weight_index) >= input_weight_synapse.starts())
        &&(weight_index < (input_weight_synapse.starts() + input_weight_synapse.interval_size()))
      ){
        is_neuron_relevant_to_weight = true;
        return false; /* no need to continue as Neuron is relevant */
      }else return true; /* kep examining the Neuron to see if it is relevant */
    });
    if(is_neuron_relevant_to_weight){ /* look through the relevant partial solutions; One Neuron shall only be part of only one partial solution. */
      for(sint32 partial_index = 0; partial_index < solution.partial_solutions_size(); ++partial_index){
        PartialSolution& partial = *solution.mutable_partial_solutions(partial_index);
        if( /* if the output of the partial solution contains the neuron */
          (static_cast<sint32>(neuron_index) >= partial.output_data().starts())
          &&(neuron_index < (partial.output_data().starts() + partial.output_data().interval_size()))
        ){ /* Iterate the inner neurons until the relevant weight synapse start is found for it */
          uint32 inner_neuron_weight_synapse_starts = 0;
          uint32 inner_neuron_weight_index_starts = 0;
          for(uint32 inner_neuron_index = 0; inner_neuron_index < partial.output_data().interval_size(); ++inner_neuron_index){
            if((partial.output_data().starts() + inner_neuron_index) == neuron_index){ /* found the relevant Neuron! */
              copy_weight_of_neuron_to_partial_solution(
                neuron_index, weight_index, inner_neuron_index, partial, inner_neuron_weight_index_starts
              );
              goto neuron_evaluation_finished; /* don't judge me */
            }/* TODO: Only copy the one weight, not the whole Neuron.. */
            for(uint32 i = 0; i < partial.weight_synapse_number(inner_neuron_index); ++i)
              inner_neuron_weight_index_starts += partial.weight_indices(inner_neuron_weight_synapse_starts + i).interval_size();
            inner_neuron_weight_synapse_starts += partial.weight_synapse_number(inner_neuron_index);
          } /* for(inner_neuron_index : every inner neuron inside the partial solution) */
        } /* if(neuron index is inside the output of the current partial solution) */
      } /* for(partial_index : every partial solution) */
    } /* if(neuron is relevant to the given weight index) */
    neuron_evaluation_finished:;
  } /* for(neuron_index : every neuron) */
}


void WeightUpdater::update_solution_with_weights(Solution& solution) const{
  sint32 partial_start_index = 0;
  while(partial_start_index < solution.partial_solutions_size()){
    if(
      (static_cast<uint32>(solution.partial_solutions_size()) < (service_context.get_max_solve_threads()/2))
      ||(solution.partial_solutions_size() < 2)
    ){
      for(sint32 partial_index = 0; partial_index < solution.partial_solutions_size(); ++partial_index){
        uint32 neuron_weight_synapse_starts = 0;
        uint32 inner_neuron_weight_index_starts = 0;
        for(uint32 inner_neuron_index = 0; inner_neuron_index < solution.partial_solutions(partial_index).output_data().interval_size(); ++inner_neuron_index){
          const uint32 neuron_index = solution.partial_solutions(partial_index).output_data().starts() + inner_neuron_index;
          copy_weights_of_neuron_to_partial_solution( /* Take over a weight from one Neuron */
            neuron_index, inner_neuron_index,
            *solution.mutable_partial_solutions(partial_index), inner_neuron_weight_index_starts
          );
          for(uint32 i = 0; i < solution.partial_solutions(partial_index).weight_synapse_number(inner_neuron_index); ++i)
            inner_neuron_weight_index_starts += solution.partial_solutions(partial_index).weight_indices(neuron_weight_synapse_starts + i).interval_size();
          neuron_weight_synapse_starts += solution.partial_solutions(partial_index).weight_synapse_number(inner_neuron_index);
        }
      }
    }else{ /* It is efficient to use multithreading */
      execution_threads.start_and_block([this, partial_start_index, &solution](uint32 thread_index){
        const sint32 partial_index = partial_start_index + thread_index;
        if(partial_index < solution.partial_solutions_size()){
          uint32 neuron_weight_synapse_starts = 0;
          uint32 inner_neuron_weight_index_starts = 0;
          for(uint32 inner_neuron_index = 0; inner_neuron_index < solution.partial_solutions(partial_index).output_data().interval_size(); ++inner_neuron_index){
            const uint32 neuron_index = solution.partial_solutions(partial_index).output_data().starts() + inner_neuron_index;
            copy_weights_of_neuron_to_partial_solution( /* Take over a weight from one Neuron */
              neuron_index, inner_neuron_index,
              *solution.mutable_partial_solutions(partial_index), inner_neuron_weight_index_starts
            );
            for(uint32 i = 0; i < solution.partial_solutions(partial_index).weight_synapse_number(inner_neuron_index); ++i)
              inner_neuron_weight_index_starts += solution.partial_solutions(partial_index).weight_indices(neuron_weight_synapse_starts + i).interval_size();
            neuron_weight_synapse_starts += solution.partial_solutions(partial_index).weight_synapse_number(inner_neuron_index);
          }
        }
      });
    }
    partial_start_index += execution_threads.get_number_of_threads();
  } /* while(partial_start_index < solution.partial_solutions_size()) */
}

void WeightUpdater::copy_weights_of_neuron_to_partial_solution(
  uint32 neuron_index, uint32 inner_neuron_index,
  PartialSolution& partial, uint32 inner_neuron_weight_index_starts
) const{ /*!Note: After shared weight optimization, this part is to be re-worked */
  uint32 weights_copied = 0;
  SynapseIterator<>::iterate(net.neuron_array(neuron_index).input_weights(),[&](
    IndexSynapseInterval weight_synapse, sint32 network_weight_index
  ){
    partial.set_weight_table(
      (inner_neuron_weight_index_starts + weights_copied), net.weight_table(network_weight_index)
    );
    ++weights_copied;
  });
}

void WeightUpdater::copy_weight_of_neuron_to_partial_solution(
  uint32 neuron_index, uint32 weight_index, uint32 inner_neuron_index,
  PartialSolution& partial, uint32 inner_neuron_weight_index_starts
) const{ /*!Note: After shared weight optimization, this part is to be re-worked */
  uint32 weights_copied = 0;
  SynapseIterator<>::iterate(net.neuron_array(neuron_index).input_weights(),[&](
    IndexSynapseInterval weight_synapse, sint32 network_weight_index
  ){
    if(static_cast<sint32>(weight_index) == network_weight_index){
      partial.set_weight_table(
        (inner_neuron_weight_index_starts + weights_copied), net.weight_table(network_weight_index)
      );
    }
    ++weights_copied;
  });
}

} /* namespace rafko_net */

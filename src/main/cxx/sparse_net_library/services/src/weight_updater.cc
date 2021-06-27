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

#include "sparse_net_library/services/weight_updater.h"

#include "sparse_net_library/services/synapse_iterator.h"

namespace sparse_net_library{

using rafko_mainframe::Service_context;

void Weight_updater::calculate_velocity_thread(const vector<sdouble32>& gradients, uint32 weight_index, uint32 weight_number){
  for(uint32 weight_iterator = 0; weight_iterator < weight_number; ++weight_iterator){
    current_velocity[weight_index + weight_iterator] = get_new_velocity(weight_index + weight_iterator, gradients);
  }
}

void Weight_updater::update_weight_with_velocity(uint32 weight_index, uint32 weight_number){
  for(uint32 weight_iterator = 0; weight_iterator < weight_number; ++weight_iterator){
    net.set_weight_table(
      weight_index + weight_iterator, get_new_weight(weight_index + weight_iterator)
    );
  }
}

void Weight_updater::calculate_velocity(const vector<sdouble32>& gradients){
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

void Weight_updater::update_weights_with_velocity(void){
  execution_threads.start_and_block([this](uint32 thread_index){
    sint32 weight_index_start = weights_to_do_in_one_thread * thread_index;
    if(weight_index_start < net.weight_table_size()){
      uint32 weight_index = (weights_to_do_in_one_thread * thread_index);
      update_weight_with_velocity(weight_index, std::min(weights_to_do_in_one_thread, (net.weight_table_size() - weight_index)));
    }
  });
}

void Weight_updater::update_solution_with_weights(Solution& solution){
  sint32 partial_start_index = 0;
  while(partial_start_index < solution.partial_solutions_size()){
    if(
      (static_cast<uint32>(solution.partial_solutions_size()) < (service_context.get_max_solve_threads()/2))
      ||(solution.partial_solutions_size() < 2)
    ){
      for(sint32 partial_index = 0; partial_index < solution.partial_solutions_size(); ++partial_index){
        uint32 neuron_weight_synapse_starts = 0;
        uint32 inner_neuron_weight_index_starts = 0;
        for(uint32 inner_neuron_index = 0; inner_neuron_index < solution.partial_solutions(partial_index).internal_neuron_number(); ++inner_neuron_index){
          const uint32 neuron_index = solution.partial_solutions(partial_index).output_data().starts() + inner_neuron_index;
          copy_weight_to_solution( /* Take over a weight from one Neuron */
            neuron_index, inner_neuron_index,
            *solution.mutable_partial_solutions(partial_index), inner_neuron_weight_index_starts
          );
          /* step forward the weight table iterator for this Neuron */
          ++inner_neuron_weight_index_starts; /* ++ for memory filter */
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
          for(uint32 inner_neuron_index = 0; inner_neuron_index < solution.partial_solutions(partial_index).internal_neuron_number(); ++inner_neuron_index){
            const uint32 neuron_index = solution.partial_solutions(partial_index).output_data().starts() + inner_neuron_index;
            copy_weight_to_solution( /* Take over a weight from one Neuron */
              neuron_index, inner_neuron_index,
              *solution.mutable_partial_solutions(partial_index), inner_neuron_weight_index_starts
            );
            /* step forward the weight table iterator for this Neuron */
            ++inner_neuron_weight_index_starts; /* ++ for memory filter */
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

void Weight_updater::copy_weight_to_solution(
  uint32 neuron_index, uint32 inner_neuron_index,
  Partial_solution& partial, uint32 inner_neuron_weight_index_starts
) const{ /*!Note: After shared weight optimization, this part is to be re-worked */
  uint32 weights_copied = 0;
  partial.set_weight_table(
    partial.memory_filter_index(inner_neuron_index),
    net.weight_table(net.neuron_array(neuron_index).memory_filter_idx())
  );
  ++weights_copied; /* ++ for Memory ratio */
  Synapse_iterator<>::iterate(net.neuron_array(neuron_index).input_weights(),[&](
    Index_synapse_interval weight_synapse, sint32 network_weight_index
  ){
    partial.set_weight_table(
      (inner_neuron_weight_index_starts + weights_copied), net.weight_table(network_weight_index)
    );
    ++weights_copied;
  });
}

} /* namespace sparse_net_library */

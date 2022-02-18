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

#include "rafko_gym/services/rafko_weight_updater.h"

#include <set>

#include "rafko_net/services/synapse_iterator.h"

namespace rafko_gym {

void RafkoWeightUpdater::update_weight_with_velocity(uint32 weight_index, uint32 weight_number){
  for(uint32 weight_iterator = 0; weight_iterator < weight_number; ++weight_iterator){
    net.set_weight_table( weight_index + weight_iterator, get_new_weight(weight_index + weight_iterator) );
  }
}

void RafkoWeightUpdater::calculate_velocity(const std::vector<sdouble32>& gradients){
  execution_threads.start_and_block([this, &gradients](uint32 thread_index){
    const uint32 weight_index_start = weights_to_do_in_one_thread * thread_index;
    const uint32 weights_to_do_in_this_thread = std::min(
      weights_to_do_in_one_thread,
      static_cast<uint32>(net.weight_table_size() - std::min(net.weight_table_size(), static_cast<sint32>(weight_index_start)))
    );
    for(uint32 weight_iterator = 0; weight_iterator < weights_to_do_in_this_thread; ++weight_iterator){
      current_velocity[weight_index_start + weight_iterator] = get_new_velocity(weight_index_start + weight_iterator, gradients);
    }
  });
}

void RafkoWeightUpdater::update_weights_with_velocity(){
  std::lock_guard<std::mutex> my_lock(reference_mutex);
  execution_threads.start_and_block([this](uint32 thread_index){
    sint32 weight_index_start = weights_to_do_in_one_thread * thread_index;
    if(weight_index_start < net.weight_table_size()){
      uint32 weight_index = (weights_to_do_in_one_thread * thread_index);
      update_weight_with_velocity(weight_index, std::min(weights_to_do_in_one_thread, (net.weight_table_size() - weight_index)));
    }
  });
}

uint32 RafkoWeightUpdater::get_relevant_partial_index_for(uint32 neuron_index) const{
  if(0 < neurons_in_partials.count(neuron_index))
    return neurons_in_partials.find(neuron_index)->second;

  for(sint32 partial_index = 0; partial_index < solution.partial_solutions_size(); ++partial_index){
    rafko_net::PartialSolution& partial = *solution.mutable_partial_solutions(partial_index);
    if( /* if the output of the partial solution contains the neuron */
      (static_cast<sint32>(neuron_index) >= partial.output_data().starts())
      &&(neuron_index < (partial.output_data().starts() + partial.output_data().interval_size()))
    ){ /* Current Neuron is part of the partial solution */
      neurons_in_partials.insert({neuron_index,partial_index});
      return partial_index;
    }
  }/*for(partial_index : solution.partial_solutions)*/

  return static_cast<uint32>(-1); /* not found! */
}

const std::vector<std::pair<uint32,uint32>>& RafkoWeightUpdater::get_relevant_partial_weight_indices_for(uint32 network_weight_index) const{
  if(0 < weights_in_partials.count(network_weight_index))
    return weights_in_partials.find(network_weight_index)->second;

  /* Find relevant Neurons for the weight, and their relative offset inside the Neuron structure */
  std::vector<std::pair<uint32,uint32>> relevant_neuron_weights; /* {Neuron index, relative_weight_index} */
  for(uint32 neuron_index = 0; static_cast<sint32>(neuron_index) < net.neuron_array_size(); ++neuron_index){
    uint32 weight_relative_index = 0u;
    /* iterate through the weights of the current neuron */
    rafko_net::SynapseIterator<>::iterate_terminatable(net.neuron_array(neuron_index).input_weights(),
    [&relevant_neuron_weights, network_weight_index, neuron_index, &weight_relative_index](uint32 weight_index){
      if(weight_index == network_weight_index){
        relevant_neuron_weights.push_back({neuron_index, weight_relative_index});
        return false; /* found the weight of the Neuron, no need to continue */
      }
      ++weight_relative_index;
      return true;
    });
  }

  /* Find the weight index value of the relevant weight inside each relevant @PartialSolution */
  std::vector<std::pair<uint32, uint32>> relevant_partials;
  std::set<uint32> found_partials;
  for(std::pair<uint32,uint32>& relevant_neural_data : relevant_neuron_weights){
    uint32 partial_index = get_relevant_partial_index_for(std::get<0>(relevant_neural_data));
    if(static_cast<sint32>(partial_index) < solution.partial_solutions_size()){ /* found a partial for the neuron! */
      if(0 == found_partials.count(partial_index)){ /* only add a partial index one time */
        found_partials.insert(partial_index);
        rafko_net::PartialSolution& partial = *solution.mutable_partial_solutions(partial_index);
        uint32 inner_neuron_weight_synapse_starts = 0;
        uint32 inner_neuron_weight_index_starts = 0;
        for(uint32 inner_neuron_index = 0; inner_neuron_index < partial.output_data().interval_size(); ++inner_neuron_index){
          if((partial.output_data().starts() + inner_neuron_index) == std::get<0>(relevant_neural_data)){ /* found the relevant Neuron! */
            uint32 inner_neuron_relative_index = 0;
            /* Iterate through the Neuron indices */
            rafko_net::SynapseIterator<>::iterate_terminatable(partial.weight_indices(),[&](sint32 inner_neuron_weight_index){
              if(std::get<1>(relevant_neural_data) == inner_neuron_relative_index){
                relevant_partials.push_back({partial_index, inner_neuron_weight_index});
                return false;
              }
              ++inner_neuron_relative_index;
              return true;
            },inner_neuron_weight_synapse_starts, partial.weight_synapse_number(inner_neuron_index));
            break; /* no need to iterate through the rest of the inner neurons.. */
          }/* if(neuron is inside the partial solution) */
          for(uint32 i = 0; i < partial.weight_synapse_number(inner_neuron_index); ++i)
            inner_neuron_weight_index_starts += partial.weight_indices(inner_neuron_weight_synapse_starts + i).interval_size();
          inner_neuron_weight_synapse_starts += partial.weight_synapse_number(inner_neuron_index);
        } /* for(inner_neuron_index : every inner neuron inside the partial solution) */
      }/* if have not added the partial yet */
    } /* If found a relevant partial */
  }/*for(all relevant neurons)*/

  /* Insert the result into the map and return */
  std::sort(relevant_partials.begin(),relevant_partials.end(),[](std::pair<uint32,uint32>& a, std::pair<uint32,uint32>& b){
    return (std::get<0>(a) > std::get<0>(b));
  });
  weights_in_partials.insert({network_weight_index, std::move(relevant_partials)});
  return weights_in_partials.find(network_weight_index)->second;
}

void RafkoWeightUpdater::update_solution_with_weight(uint32 weight_index) const{
  std::lock_guard<std::mutex> my_lock(reference_mutex);
  assert(static_cast<sint32>(weight_index) < net.weight_table_size());
  const std::vector<std::pair<uint32, uint32>>& relevant_partial_weights = get_relevant_partial_weight_indices_for(weight_index);
  for(const std::pair<uint32,uint32>& relevant_partial_weight : relevant_partial_weights){
    solution.mutable_partial_solutions(std::get<0>(relevant_partial_weight))->set_weight_table(
      std::get<1>(relevant_partial_weight), net.weight_table(weight_index)
    );
  }
}

void RafkoWeightUpdater::update_solution_with_weights() const{
  std::lock_guard<std::mutex> my_lock(reference_mutex);
  sint32 partial_start_index = 0;
  while(partial_start_index < solution.partial_solutions_size()){
    if(
      (static_cast<uint32>(solution.partial_solutions_size()) < (settings.get_max_solve_threads()/2))
      ||(solution.partial_solutions_size() < 2)
    ){
      for(sint32 partial_index = 0; partial_index < solution.partial_solutions_size(); ++partial_index){
        uint32 neuron_weight_synapse_starts = 0;
        uint32 inner_neuron_weight_index_starts = 0;
        for(uint32 inner_neuron_index = 0; inner_neuron_index < solution.partial_solutions(partial_index).output_data().interval_size(); ++inner_neuron_index){
          const uint32 neuron_index = solution.partial_solutions(partial_index).output_data().starts() + inner_neuron_index;
          copy_weights_of_neuron_to_partial_solution( /* Take over a weight from one Neuron */
            neuron_index, *solution.mutable_partial_solutions(partial_index), inner_neuron_weight_index_starts
          );
          for(uint32 i = 0; i < solution.partial_solutions(partial_index).weight_synapse_number(inner_neuron_index); ++i)
            inner_neuron_weight_index_starts += solution.partial_solutions(partial_index).weight_indices(neuron_weight_synapse_starts + i).interval_size();
          neuron_weight_synapse_starts += solution.partial_solutions(partial_index).weight_synapse_number(inner_neuron_index);
        }
      }
    }else{ /* It is efficient to use multithreading */
      execution_threads.start_and_block([this, partial_start_index](uint32 thread_index){
        const sint32 partial_index = partial_start_index + thread_index;
        if(partial_index < solution.partial_solutions_size()){
          uint32 neuron_weight_synapse_starts = 0;
          uint32 inner_neuron_weight_index_starts = 0;
          for(uint32 inner_neuron_index = 0; inner_neuron_index < solution.partial_solutions(partial_index).output_data().interval_size(); ++inner_neuron_index){
            const uint32 neuron_index = solution.partial_solutions(partial_index).output_data().starts() + inner_neuron_index;
            copy_weights_of_neuron_to_partial_solution( /* Take over a weight from one Neuron */
              neuron_index, *solution.mutable_partial_solutions(partial_index), inner_neuron_weight_index_starts
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

void RafkoWeightUpdater::copy_weights_of_neuron_to_partial_solution(
  uint32 neuron_index, rafko_net::PartialSolution& partial, uint32 inner_neuron_weight_index_starts
) const{ /*!Note: After shared weight optimization, this part is to be re-worked */
  uint32 weights_copied = 0;
  rafko_net::SynapseIterator<>::iterate(net.neuron_array(neuron_index).input_weights(),[&](sint32 network_weight_index){
    partial.set_weight_table(
      (inner_neuron_weight_index_starts + weights_copied), net.weight_table(network_weight_index)
    );
    ++weights_copied;
  });
}

} /* namespace rafko_gym */

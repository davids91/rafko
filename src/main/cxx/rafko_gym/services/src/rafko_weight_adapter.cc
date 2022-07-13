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

#include "rafko_gym/services/rafko_weight_adapter.hpp"

#include <set>

#include "rafko_net/services/synapse_iterator.hpp"
#include "rafko_mainframe/services/rafko_assertion_logger.hpp"

namespace rafko_gym {

std::uint32_t RafkoWeightAdapter::get_relevant_partial_index_for(
  std::uint32_t neuron_index, const rafko_net::Solution& solution,
  std::unordered_map<std::uint32_t, std::uint32_t>& neurons_in_partials
){
  std::unordered_map<std::uint32_t, std::uint32_t>::iterator iter = neurons_in_partials.find(neuron_index);
  if(iter != neurons_in_partials.end())
    return iter->second;

  for(std::int32_t partial_index = 0; partial_index < solution.partial_solutions_size(); ++partial_index){
    const rafko_net::PartialSolution& partial = solution.partial_solutions(partial_index);
    if( /* if the output of the partial solution contains the neuron */
      (static_cast<std::int32_t>(neuron_index) >= partial.output_data().starts())
      &&(neuron_index < (partial.output_data().starts() + partial.output_data().interval_size()))
    ){ /* Current Neuron is part of the partial solution */
      /* add every neuron index into partial index cache */
      for(std::uint32_t i = 0; i < partial.output_data().interval_size(); ++i)
        neurons_in_partials.insert({(partial.output_data().starts() + i),partial_index});
      return partial_index;
    }
  }/*for(partial_index : solution.partial_solutions)*/

  return static_cast<std::uint32_t>(-1); /* not found! */
}

PartialWeightPairs& RafkoWeightAdapter::get_relevant_partial_weight_indices_for(std::uint32_t network_weight_index) const{
  std::unordered_map<std::uint32_t, std::vector<std::pair<std::uint32_t,std::uint32_t>>>::iterator iter = weights_in_partials.find(network_weight_index);
  if(iter != weights_in_partials.end())
    return iter->second;

  /* Find relevant Neurons for the weight, and their relative offset inside the Neuron structure */
  std::vector<std::pair<std::uint32_t,std::uint32_t>> relevant_neuron_weights; /* {Neuron index, relative_weight_index} */
  for(std::uint32_t neuron_index = 0; static_cast<std::int32_t>(neuron_index) < net.neuron_array_size(); ++neuron_index){
    std::uint32_t weight_relative_index = 0u;
    /* iterate through the weights of the current neuron */
    rafko_net::SynapseIterator<>::iterate_terminatable(net.neuron_array(neuron_index).input_weights(),
    [&relevant_neuron_weights, network_weight_index, neuron_index, &weight_relative_index](std::uint32_t weight_index){
      if(weight_index == network_weight_index){
        relevant_neuron_weights.push_back({neuron_index, weight_relative_index});
        return false; /* found the weight of the Neuron, no need to continue */
      }
      ++weight_relative_index;
      return true;
    });
  }

  /* Find the weight index value of the relevant weight inside each relevant @PartialSolution */
  std::vector<std::pair<std::uint32_t, std::uint32_t>> relevant_partials;
  std::set<std::uint32_t> found_partials;
  for(std::pair<std::uint32_t,std::uint32_t>& relevant_neural_data : relevant_neuron_weights){
    std::uint32_t partial_index = get_relevant_partial_index_for(std::get<0>(relevant_neural_data));
    if(static_cast<std::int32_t>(partial_index) < solution.partial_solutions_size()){ /* found a partial for the neuron! */
      if(0 == found_partials.count(partial_index)){ /* only add a partial index one time */
        found_partials.insert(partial_index);
        rafko_net::PartialSolution& partial = *solution.mutable_partial_solutions(partial_index);
        std::uint32_t inner_neuron_weight_synapse_starts = 0;
        std::uint32_t inner_neuron_weight_index_starts = 0;
        for(std::uint32_t inner_neuron_index = 0; inner_neuron_index < partial.output_data().interval_size(); ++inner_neuron_index){
          if((partial.output_data().starts() + inner_neuron_index) == std::get<0>(relevant_neural_data)){ /* found the relevant Neuron! */
            std::uint32_t inner_neuron_relative_index = 0;
            /* Iterate through the Neuron indices */
            rafko_net::SynapseIterator<>::iterate_terminatable(partial.weight_indices(),[&](std::int32_t inner_neuron_weight_index){
              if(std::get<1>(relevant_neural_data) == inner_neuron_relative_index){
                relevant_partials.push_back({partial_index, inner_neuron_weight_index});
                return false;
              }
              ++inner_neuron_relative_index;
              return true;
            },inner_neuron_weight_synapse_starts, partial.weight_synapse_number(inner_neuron_index));
            break; /* no need to iterate through the rest of the inner neurons.. */
          }/* if(neuron is inside the partial solution) */
          for(std::uint32_t i = 0; i < partial.weight_synapse_number(inner_neuron_index); ++i)
            inner_neuron_weight_index_starts += partial.weight_indices(inner_neuron_weight_synapse_starts + i).interval_size();
          inner_neuron_weight_synapse_starts += partial.weight_synapse_number(inner_neuron_index);
        } /* for(inner_neuron_index : every inner neuron inside the partial solution) */
      }/* if have not added the partial yet */
    } /* If found a relevant partial */
  }/*for(all relevant neurons)*/

  /* Insert the result into the map and return */
  std::sort(relevant_partials.begin(),relevant_partials.end(),[](std::pair<std::uint32_t,std::uint32_t>& a, std::pair<std::uint32_t,std::uint32_t>& b){
    return (std::get<0>(a) > std::get<0>(b));
  });
  weights_in_partials.insert({network_weight_index, std::move(relevant_partials)});
  return weights_in_partials.find(network_weight_index)->second;
}

#if(RAFKO_USES_OPENCL)
std::uint32_t RafkoWeightAdapter::get_device_weight_table_start_for(
  std::uint32_t partial_index, const rafko_net::Solution& solution,
  std::unordered_map<std::uint32_t, std::uint32_t>& weight_starts_in_partials
){
  RFASSERT( static_cast<std::int32_t>(partial_index) < solution.partial_solutions_size() );

  std::unordered_map<std::uint32_t, std::uint32_t>::iterator iter = weight_starts_in_partials.find(partial_index);
  if(iter != weight_starts_in_partials.end())
    return iter->second;

  /* find weight start of partial */
  std::uint32_t weight_start = 0u;
  for(std::uint32_t p = 0u; p <= partial_index; ++p){
    weight_starts_in_partials.insert({p, weight_start});
    weight_start += solution.partial_solutions(p).weight_table_size();
  }
  return (weight_start - solution.partial_solutions(partial_index).weight_table_size());
}
#endif/*(RAFKO_USES_OPENCL)*/

std::uint32_t RafkoWeightAdapter::get_weight_synapse_start_index_in_partial(
  std::uint32_t neuron_index, const rafko_net::PartialSolution& partial,
  std::unordered_map<std::uint32_t, std::uint32_t>& weight_synapse_starts_in_partial
){
  std::uint32_t partial_first_neuron_index = partial.output_data().starts();
  std::uint32_t partial_neuron_number = partial.output_data().interval_size();
  RFASSERT( neuron_index >= partial_first_neuron_index );
  RFASSERT( neuron_index < (partial_first_neuron_index + partial_neuron_number) );

  std::unordered_map<std::uint32_t, std::uint32_t>::iterator iter = weight_synapse_starts_in_partial.find(neuron_index);
  if(iter != weight_synapse_starts_in_partial.end())
    return iter->second;

  /* find weight synapse start of partial */
  std::uint32_t weight_synapse_start = 0u;
  std::uint32_t inner_neuron_index = neuron_index - partial_first_neuron_index;
  for(std::uint32_t i = 0u; i <= inner_neuron_index; ++i){
    weight_synapse_starts_in_partial.insert({i, weight_synapse_start});
    weight_synapse_start += partial.weight_synapse_number(i);
  }
  return (weight_synapse_start - partial.weight_synapse_number(inner_neuron_index));
}

void RafkoWeightAdapter::update_solution_with_weight(std::uint32_t weight_index) const{
  std::lock_guard<std::mutex> my_lock(reference_mutex);
  RFASSERT(static_cast<std::int32_t>(weight_index) < net.weight_table_size());
  const std::vector<std::pair<std::uint32_t, std::uint32_t>>& relevant_partial_weights = get_relevant_partial_weight_indices_for(weight_index);
  for(const std::pair<std::uint32_t,std::uint32_t>& relevant_partial_weight : relevant_partial_weights){
    solution.mutable_partial_solutions(std::get<0>(relevant_partial_weight))->set_weight_table(
      std::get<1>(relevant_partial_weight), net.weight_table(weight_index)
    );
  }
}

void RafkoWeightAdapter::update_solution_with_weights() const{
  std::lock_guard<std::mutex> my_lock(reference_mutex);
  std::int32_t partial_start_index = 0;
  while(partial_start_index < solution.partial_solutions_size()){
    if(
      (static_cast<std::uint32_t>(solution.partial_solutions_size()) < (settings.get_max_solve_threads()/2))
      ||(solution.partial_solutions_size() < 2)
    ){
      for(std::int32_t partial_index = 0; partial_index < solution.partial_solutions_size(); ++partial_index){
        std::uint32_t neuron_weight_synapse_starts = 0;
        std::uint32_t inner_neuron_weight_index_starts = 0;
        for(std::uint32_t inner_neuron_index = 0; inner_neuron_index < solution.partial_solutions(partial_index).output_data().interval_size(); ++inner_neuron_index){
          const std::uint32_t neuron_index = solution.partial_solutions(partial_index).output_data().starts() + inner_neuron_index;
          copy_weights_of_neuron_to_partial_solution( /* Take over a weight from one Neuron */
            neuron_index, *solution.mutable_partial_solutions(partial_index), inner_neuron_weight_index_starts
          );
          for(std::uint32_t i = 0; i < solution.partial_solutions(partial_index).weight_synapse_number(inner_neuron_index); ++i)
            inner_neuron_weight_index_starts += solution.partial_solutions(partial_index).weight_indices(neuron_weight_synapse_starts + i).interval_size();
          neuron_weight_synapse_starts += solution.partial_solutions(partial_index).weight_synapse_number(inner_neuron_index);
        }
      }
    }else{ /* It is efficient to use multithreading */
      execution_threads.start_and_block([this, partial_start_index](std::uint32_t thread_index){
        const std::int32_t partial_index = partial_start_index + thread_index;
        if(partial_index < solution.partial_solutions_size()){
          std::uint32_t neuron_weight_synapse_starts = 0;
          std::uint32_t inner_neuron_weight_index_starts = 0;
          for(std::uint32_t inner_neuron_index = 0; inner_neuron_index < solution.partial_solutions(partial_index).output_data().interval_size(); ++inner_neuron_index){
            const std::uint32_t neuron_index = solution.partial_solutions(partial_index).output_data().starts() + inner_neuron_index;
            copy_weights_of_neuron_to_partial_solution( /* Take over a weight from one Neuron */
              neuron_index, *solution.mutable_partial_solutions(partial_index), inner_neuron_weight_index_starts
            );
            for(std::uint32_t i = 0; i < solution.partial_solutions(partial_index).weight_synapse_number(inner_neuron_index); ++i)
              inner_neuron_weight_index_starts += solution.partial_solutions(partial_index).weight_indices(neuron_weight_synapse_starts + i).interval_size();
            neuron_weight_synapse_starts += solution.partial_solutions(partial_index).weight_synapse_number(inner_neuron_index);
          }
        }
      });
    }
    partial_start_index += execution_threads.get_number_of_threads();
  } /* while(partial_start_index < solution.partial_solutions_size()) */
}

void RafkoWeightAdapter::copy_weights_of_neuron_to_partial_solution(
  std::uint32_t neuron_index, rafko_net::PartialSolution& partial, std::uint32_t inner_neuron_weight_index_starts
) const{ /*!Note: After shared weight optimization, this part is to be re-worked */
  std::uint32_t weights_copied = 0;
  rafko_net::SynapseIterator<>::iterate(net.neuron_array(neuron_index).input_weights(),[&](std::int32_t network_weight_index){
    partial.set_weight_table(
      (inner_neuron_weight_index_starts + weights_copied), net.weight_table(network_weight_index)
    );
    ++weights_copied;
  });
}

} /* namespace rafko_gym */

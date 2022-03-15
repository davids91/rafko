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
#include "rafko_gym/models/rafko_network_delta_chain_link.h"

#include <limits>
#include <math.h>

#include "rafko_net/services/synapse_iterator.h"

namespace rafko_gym{

rafko_net::RafkoNet RafkoNetworkDeltaChainLink::get_current_network(){
  if(network_built) return network;

  if(parent){
    network = parent->get_current_network();
  }else{
    network = original_network;
  }

  apply_to_network(data, network);
  return network;
}

void RafkoNetworkDeltaChainLink::apply_to_network(NetworkDeltaChainLinkData& delta, rafko_net::RafkoNet& network){
  if(0u == (delta.simple_changes_size() + delta.structural_changes_size())) return; /* No changes == nothing to do */
  std::int32_t simple_changes_index = 0u;
  std::int32_t structural_changes_index = 0u;
  while(
    (simple_changes_index < delta.simple_changes_size())
    &&(structural_changes_index < delta.structural_changes_size())
  ){
    if(
      (structural_changes_index < delta.structural_changes_size())
      &&(
        (simple_changes_index >= delta.simple_changes_size())
        ||(delta.structural_changes(structural_changes_index).version() < delta.simple_changes(simple_changes_index).version())
      )
    ){ /* apply the structural change */
      /*TODO: Actually apply the structural change */
      ++structural_changes_index;
    }else{ /* apply the simple change */
      /*TODO: Actually apply the simple change */
      apply_change(delta.simple_changes(simple_changes_index), network);
      ++simple_changes_index;
    }
  }/*while(both change iterators ar inside bounds)*/
}

void RafkoNetworkDeltaChainLink::apply_change(const NonStructuralNetworkDelta& change, rafko_net::RafkoNet& network){
  /* Apply weight changes */
  std::uint32_t change_value_index = 0;
  if(1u == change.weights_delta().weight_synapses_size()){
    std::copy(
      change.weights_delta().values().begin(),
      change.weights_delta().values().begin() + change.weights_delta().weight_synapses(0).interval_size(),
      network.mutable_weight_table()->begin() + change.weights_delta().weight_synapses(0).starts()
    );
  }else{
    rafko_net::SynapseIterator<>::iterate(change.weights_delta().weight_synapses(),
    [&network, &change_value_index, &change](std::int32_t weight_index){
      network.set_weight_table(
        weight_index, (network.weight_table(weight_index) + change.weights_delta().values(change_value_index))
      );
      ++change_value_index;
    });
  }

  /* Apply functional changes */
  rafko_net::SynapseIterator<> relevant_neurons_iterator(change.functional_delta().relevant_neurons());
  RFASSERT(change.functional_delta().input_functions_size() == change.functional_delta().transfer_functions_size());
  RFASSERT(change.functional_delta().transfer_functions_size() == change.functional_delta().spike_functions_size());
  RFASSERT(change.functional_delta().spike_functions_size() == static_cast<std::int32_t>(relevant_neurons_iterator.size()));
  change_value_index = 0;
  relevant_neurons_iterator.iterate([&network, &change_value_index, &change](std::int32_t neuron_index){
    network.mutable_neuron_array(neuron_index)->set_input_function(change.functional_delta().input_functions(change_value_index));
    network.mutable_neuron_array(neuron_index)->set_transfer_function(change.functional_delta().transfer_functions(change_value_index));
    network.mutable_neuron_array(neuron_index)->set_spike_function(change.functional_delta().spike_functions(change_value_index));
    ++change_value_index;
  });
}

} /* namespace rafko_gym */

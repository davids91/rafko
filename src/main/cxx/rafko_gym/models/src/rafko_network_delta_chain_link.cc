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

rafko_utilities::DataPool<> RafkoNetworkDeltaChainLink::tmp_data_pool(8u,100u);
/*!Note: Maximum number of vectors to be allocated through here is capped at the theoretical number of
 * threads optimal to any system, which is assumed to be 8 based on the assumption that the processor
 * running the library has 8 cores.
 * Unfortunately there is no good guess as to what is the number of weights to be used with this
 * system, as a network can have ANY number of weights; So 100 is estimated to fit most cases;
 * This optimization is not believed to have a huge impact.
 */

rafko_net::RafkoNet RafkoNetworkDeltaChainLink::get_current_network(){
  if(network_built) return current_network;

  if(parent){
    current_network = parent->get_current_network();
  }else{
    current_network = original_network;
  }

  apply_to_network(data, current_network);

  network_built = true;
  network_structure_built = true;
  return current_network;
}

void RafkoNetworkDeltaChainLink::store_change(std::uint32_t weight_index, double weight_delta){
  NetworkWeightVectorDelta& weights_delta = *data.mutable_simple_changes(data.simple_changes_size() - 1)->mutable_weights_delta();
  std::uint32_t values_index = 0;
  std::uint32_t values_index_target = weights_delta.values_size();
  std::uint32_t weight_synapse_index_target = weights_delta.weight_synapses_size();
  rafko_net::IndexSynapseInterval tmp_synapse_interval;

  for(std::uint32_t weight_syn_index = 0; static_cast<std::int32_t>(weight_syn_index) < weights_delta.weight_synapses_size(); ++weight_syn_index){
    if( /* If the weight synapse is at or in-between the first index before the start of the synapse.. */
      (
        ((0 < weights_delta.weight_synapses(weight_syn_index).starts())
        &&( weights_delta.weight_synapses(weight_syn_index).starts()-1u) <= weight_index )
        ||( 0 == weights_delta.weight_synapses(weight_syn_index).starts() )
      )&&( /* ..and the one after the last index */
        (weights_delta.weight_synapses(weight_syn_index).starts() + weights_delta.weight_synapses(weight_syn_index).interval_size())
        >= weight_index
      )
    ){ /* current weight synapse is a sitable target to place the current fragment in */
      weight_synapse_index_target = weight_syn_index;
      values_index_target = values_index;
      break; /* Found a suitable synapse, no need to continue */
    }
    values_index += weights_delta.weight_synapses(weight_syn_index).interval_size();
  } /* Go through the synapses saving the last place */
  if(
    (0 == weights_delta.weight_synapses_size())
    ||(static_cast<std::int32_t>(weight_synapse_index_target) >= weights_delta.weight_synapses_size())
    ||(static_cast<std::int32_t>(values_index_target) >= weights_delta.values_size())
  ){
    weights_delta.add_values(weight_delta);
    tmp_synapse_interval.set_interval_size(1);
    tmp_synapse_interval.set_starts(weight_index);
    *weights_delta.add_weight_synapses() = tmp_synapse_interval;
  }else{
    const std::uint32_t synapse_starts = weights_delta.weight_synapses(weight_synapse_index_target).starts();
    const std::uint32_t synapse_size = weights_delta.weight_synapses(weight_synapse_index_target).interval_size();
    const std::uint32_t synapse_ends = synapse_starts + synapse_size;

    if(
      (0 < synapse_starts) /* Synapse doesn't start at 0 */
      &&((synapse_starts-1) == weight_index) /* And the weight index points to the first index before the synapse */
    ){
      weights_delta.mutable_weight_synapses(weight_synapse_index_target)->set_interval_size(synapse_size + 1);
      weights_delta.mutable_weight_synapses(weight_synapse_index_target)->set_starts(synapse_starts - 1);
      insert_element_at_position(*weights_delta.mutable_values(),weight_delta,values_index_target);
    }else if(
      (synapse_starts <= weight_index)
      &&(synapse_ends > weight_index)
    ){ /* the index is inside the synapse */
      weights_delta.set_values(
        values_index_target + weight_index - synapse_starts,
        weights_delta.values(values_index_target + weight_index - synapse_starts) + weight_delta
      );
    }else{ /* The index is the first index after the synapse */
      weights_delta.mutable_weight_synapses(weight_synapse_index_target)->set_interval_size(synapse_size + 1);
      insert_element_at_position( *weights_delta.mutable_values(), weight_delta, (values_index_target + synapse_size) );
    }
  }
  network_built = false;
}

void RafkoNetworkDeltaChainLink::store_change(std::vector<double>& weight_delta){
  if(!network_structure_built)
    current_network = get_current_network();

  RFASSERT( current_network.weight_table_size() == static_cast<std::int32_t>(weight_delta.size()) );

  NetworkWeightVectorDelta& current_weights_delta = *data.mutable_simple_changes(data.simple_changes_size() - 1)->mutable_weights_delta();

  /* Unwrap current delta to a continous array */
  std::vector<double>& tmp_delta_array = tmp_data_pool.reserve_buffer(current_network.weight_table_size());
  std::fill(tmp_delta_array.begin(), tmp_delta_array.end(), 0.0);
  unwrap_change_to(tmp_delta_array, current_weights_delta);

  /* add weight delta to array */
  std::transform(
    current_weights_delta.values().begin(), current_weights_delta.values().end(),
    tmp_delta_array.begin(), tmp_delta_array.begin(),
    [](const double& a, const double& b){ return a + b; }
  );

  /* set weight delta to current array */
  current_weights_delta.mutable_weight_synapses()->Clear();
  current_weights_delta.add_weight_synapses()->set_starts(0);
  current_weights_delta.mutable_weight_synapses(0)->set_interval_size(weight_delta.size());
  current_weights_delta.mutable_values()->Assign(weight_delta.begin(),weight_delta.end());

  network_built = false;
}

void RafkoNetworkDeltaChainLink::store_change(NetworkWeightVectorDelta&& weight_delta){
  if(!network_structure_built)
    current_network = get_current_network();

  std::vector<double>& tmp_delta_array = tmp_data_pool.reserve_buffer(current_network.weight_table_size());

  /* Unwrap current */
  /* Unwrap given */
  /* Store non-zero weights */
}

void RafkoNetworkDeltaChainLink::store_change(NetworkDeltaChainLinkData&& weight_delta){
  /* Add to top of the data */

  if(0 < data.structural_changes_size()) /* only when structural changes are present */
    network_structure_built = false; /* should the network structure be invalidated */
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

void RafkoNetworkDeltaChainLink::unwrap_change_to(std::vector<double>& vector, NetworkWeightVectorDelta& delta){
  rafko_net::SynapseIterator<> weights_iterator(delta.weight_synapses());

  if(
    (1u == delta.weight_synapses_size())&&(0u == delta.weight_synapses(0).starts())
    &&(vector.size() == delta.weight_synapses(0).interval_size())
  ){/* if the whole delta is stored in a full continous array */
    vector = { delta.values().begin(), delta.values().end() };
  }else if(1u == delta.weight_synapses_size()){
    RFASSERT( (delta.weight_synapses(0).starts() + delta.weight_synapses(0).interval_size()) <= vector.size() );
    std::copy( delta.values().begin(), delta.values().end(), vector.begin() + delta.weight_synapses(0).starts() );
  }else{
    std::uint32_t values_index = 0u;
    weights_iterator.iterate([&delta, &vector, &values_index](std::uint32_t weight_index){
      vector[weight_index] = delta.values(values_index);
      ++values_index;
    });
  }
}

} /* namespace rafko_gym */

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
#include <thread>
#include <map>
#include <utility>

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

  if(parent) current_network = parent->get_current_network();
    else current_network = original_network;

  apply_to_network(data, current_network);

  network_built = true;
  network_structure_built = true;
  return current_network;
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
  if(is_last_change_simple()){
    if(!network_structure_built)
      current_network = get_current_network();

    std::vector<double>& tmp_delta_array = tmp_data_pool.reserve_buffer(current_network.weight_table_size());
    std::vector<double>& tmp_delta_array2 = tmp_data_pool.reserve_buffer(current_network.weight_table_size());
    std::fill(tmp_delta_array.begin(),tmp_delta_array.end(), 0.0);
    std::fill(tmp_delta_array2.begin(),tmp_delta_array2.end(), 0.0);

    unwrap_change_to(tmp_delta_array, data.simple_changes(data.simple_changes_size() - 1u).weights_delta());
    unwrap_change_to(tmp_delta_array2, weight_delta);
    std::transform(
      tmp_delta_array2.begin(), tmp_delta_array2.end(), tmp_delta_array.begin(), tmp_delta_array.begin(),
      [](const double& a, const double& b){ return a + b; }
    );
    tmp_data_pool.release_buffer(tmp_delta_array2);
    for(std::uint32_t weight_index = 0; weight_index < tmp_delta_array.size(); ++weight_index){
      if(0.0 < tmp_delta_array[weight_index]){
        store_change(weight_index, tmp_delta_array[weight_index]);
      }
    }
    tmp_data_pool.release_buffer(tmp_delta_array);
  }else{ /* unable to update a currently available weight delta, so add it into the vector */
    data.add_simple_changes()->set_allocated_weights_delta(&weight_delta);
  }

  data.mutable_simple_changes(data.simple_changes_size() - 1u)->set_version(get_latest_version() + 1u);
  network_built = false;
}

void RafkoNetworkDeltaChainLink::store_change(NonStructuralNetworkDelta&& change){
  { /* store the weights vector delta change */
    std::unique_ptr<NetworkWeightVectorDelta> weights_delta(change.release_weights_delta());
    if(weights_delta)
      store_change(std::move(*weights_delta));
    /*!Note: this guarantees that the last change is a non-structural one; It eiser uses it or adds another */
    /*!Note: this also guarantees that the network is not built */
  }
  std::thread make_network_thread([this](){
    current_network = get_current_network();
  });

  /* store the provided functional changes */
  NetworkNeuronFunctionDelta& current_functional_changes = *data.mutable_simple_changes(data.simple_changes_size() - 1u)->mutable_functional_delta();
  rafko_net::SynapseIterator<> relevant_neurons_iterator(current_functional_changes.relevant_neurons());
  RFASSERT(change.functional_delta().input_functions_size() == change.functional_delta().transfer_functions_size());
  RFASSERT(change.functional_delta().transfer_functions_size() == change.functional_delta().spike_functions_size());
  RFASSERT(change.functional_delta().spike_functions_size() == static_cast<std::int32_t>(relevant_neurons_iterator.size()));

  std::uint32_t change_value_index = 0; /* store the current changes in a map */
  std::map<std::uint32_t, std::tuple<rafko_net::Input_functions, rafko_net::Transfer_functions, rafko_net::Spike_functions>> neuron_functional_changes;
  relevant_neurons_iterator.iterate([&neuron_functional_changes, &current_functional_changes, &change_value_index](std::int32_t neuron_index){
    neuron_functional_changes.insert_or_assign(neuron_index, std::make_tuple(
      current_functional_changes.input_functions(change_value_index),
      current_functional_changes.transfer_functions(change_value_index),
      current_functional_changes.spike_functions(change_value_index)
    ));
    ++change_value_index;
  });

  /* store provided changes in the same map, overwriting previous values on collision */
  rafko_net::SynapseIterator<> also_relevant_neurons_iterator = rafko_net::SynapseIterator<>(change.functional_delta().relevant_neurons());
  RFASSERT(change.functional_delta().input_functions_size() == change.functional_delta().transfer_functions_size());
  RFASSERT(change.functional_delta().transfer_functions_size() == change.functional_delta().spike_functions_size());
  RFASSERT(change.functional_delta().spike_functions_size() == static_cast<std::int32_t>(also_relevant_neurons_iterator.size()));
  change_value_index = 0u;
  also_relevant_neurons_iterator.iterate([&neuron_functional_changes, &change, &change_value_index](std::int32_t neuron_index){
    neuron_functional_changes.insert_or_assign(neuron_index, std::make_tuple(
      change.functional_delta().input_functions(change_value_index),
      change.functional_delta().transfer_functions(change_value_index),
      change.functional_delta().spike_functions(change_value_index)
    ));
    ++change_value_index;
  });

  while(true){
    if(make_network_thread.joinable()){
      make_network_thread.join();
      break;
    }
  }


  /* Go through the assembled changes and store whichever is an actual change */
  current_functional_changes.mutable_relevant_neurons()->Clear(); /* functional changes are being rebuilt from scratch */
  rafko_net::IndexSynapseInterval* latest_synapse = nullptr;
  for(auto const& [neuron_index, functional_change] : neuron_functional_changes){
    /*!Note: This logic relies heavily on the fact that the elements inside a std::map are stored in an increasing order by key ( which is Neuron index in this case) */
    if(
      (current_network.neuron_array(neuron_index).input_function() != std::get<rafko_net::Input_functions>(functional_change))
      ||(current_network.neuron_array(neuron_index).transfer_function() != std::get<rafko_net::Transfer_functions>(functional_change))
      ||(current_network.neuron_array(neuron_index).spike_function() != std::get<rafko_net::Spike_functions>(functional_change))
    ){ /* the change is actually modifying a current network */
      if( /* in case there are no synapses yet.. */
        (nullptr == latest_synapse)
        ||(0u == current_functional_changes.relevant_neurons_size())
        ||((latest_synapse->starts() + latest_synapse->interval_size()) != neuron_index)
      ){ /* or the current index is not continuing the latest synapse */
        latest_synapse = current_functional_changes.add_relevant_neurons();
        latest_synapse->set_interval_size(1u);
        latest_synapse->set_starts(neuron_index);
      }else{ /* in case the current change is continuing the latest synapse */
        latest_synapse->set_interval_size(latest_synapse->interval_size() + 1u);
      }
      current_functional_changes.add_input_functions(std::get<rafko_net::Input_functions>(functional_change));
      current_functional_changes.add_transfer_functions(std::get<rafko_net::Transfer_functions>(functional_change));
      current_functional_changes.add_spike_functions(std::get<rafko_net::Spike_functions>(functional_change));
    }/*if(current values of functional_change modify anything at all in a current network)*/
  }/*for(every {neuron_index, functional change in neuron_functionl_changes})*/
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

void RafkoNetworkDeltaChainLink::apply_change(std::uint32_t weight_index, double weight_delta, NetworkWeightVectorDelta& weights_delta){
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
}

void RafkoNetworkDeltaChainLink::unwrap_change_to(std::vector<double>& vector, const NetworkWeightVectorDelta& delta){
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
    rafko_net::SynapseIterator<>::iterate(delta.weight_synapses(),
    [&delta, &vector, &values_index](std::uint32_t weight_index){
      vector[weight_index] = delta.values(values_index);
      ++values_index;
    });
  }
}

} /* namespace rafko_gym */

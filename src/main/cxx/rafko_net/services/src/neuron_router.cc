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

#include "rafko_net/services/neuron_router.h"

#include <algorithm>
#include <thread>

#include "rafko_mainframe/services/rafko_assertion_logger.h"
#include "rafko_net/models/neuron_info.h"

#include "rafko_net/services/synapse_iterator.h"

namespace rafko_net{

NeuronRouter::NeuronRouter(const RafkoNet& rafko_net)
: net(rafko_net)
, output_layer_iterator(net.neuron_array_size() - net.output_neuron_number())
, neuron_number_of_inputs(net.neuron_array_size())
, features_assigned_to_neurons(net.neuron_array_size())
{
  for(int neuron_iterator = 0; neuron_iterator < net.neuron_array_size(); ++neuron_iterator) {
      for(int synapse_iterator = 0;
        synapse_iterator < net.neuron_array(neuron_iterator).input_indices_size();
        ++synapse_iterator
      ) neuron_number_of_inputs[neuron_iterator] += net.neuron_array(neuron_iterator).input_indices(synapse_iterator).interval_size();
      neuron_states.push_back(std::make_unique<std::atomic<std::uint32_t>>());
  } /* Calculating how many children one Neuron has */

  for(std::int32_t feature_index = 0; feature_index < rafko_net.neuron_group_features_size(); feature_index++){
    /* For each relevant feature group */
    const FeatureGroup& feature_group = rafko_net.neuron_group_features(feature_index);
    tracked_features.push_back( FeatureGroupCache(net, feature_index) );
    SynapseIterator<>::iterate(feature_group.relevant_neurons(),[&](std::int32_t neuron_index){
      features_assigned_to_neurons[neuron_index].push_back(tracked_features.size() - 1u);
    });
    /*!Note: tracked_features should be unchanged after constructor as it is used by index in the construction logic */
  }/* for(each feature group in the network) */
}

void NeuronRouter::collect_subset(std::uint8_t arg_max_solve_threads, double arg_device_max_megabytes, bool strict){
  collection_running = true;
  std::vector<std::thread> processing_threads;
  for(std::uint8_t thread_iterator = 0; thread_iterator < arg_max_solve_threads; thread_iterator++){
    processing_threads.push_back(
      std::thread(
        &NeuronRouter::collect_subset_thread,
        this, arg_max_solve_threads, arg_device_max_megabytes, thread_iterator, strict
      )
    ); /* Add threads for processing */
  }

  std::for_each(processing_threads.begin(),processing_threads.end(),[](std::thread& processing_thread){
    if(true == processing_thread.joinable())processing_thread.join();
  });
  if(strict)std::sort(net_subset.begin(),net_subset.end());
  collection_running = false;
  ++iteration;
}

void NeuronRouter::collect_subset_thread(std::uint8_t arg_max_solve_threads, double arg_device_max_megabytes, std::uint8_t thread_index, bool strict){
  /** @brief visiting:
   * In order of the iteration, the visited neuron indexes. The First Index is always one of the Output Layer Neurons
   */
  std::vector<std::uint32_t> visiting(1,
    (output_layer_iterator + ((net.neuron_array_size()-1-output_layer_iterator)/arg_max_solve_threads)*thread_index)
  ); /* The first Neuron to be visited is decided based on the number of threads, to make sure the threads are as independent as possible */
  std::uint32_t visiting_next = visiting.back();

  while( /* Iterate the Net until every possible Neuron is collected into an independent subset of it */
    (net.neuron_array_size() > static_cast<int>(visiting.back())) /* The currently visiting Neuron is inside bounds of the net */
    &&(static_cast<int>(output_layer_iterator) < net.neuron_array_size()) /* Until the whole output layer is processed */
    &&(net_subset_size_bytes/* Bytes *// (1024.0) /* KB *// (1024.0) /* MB */ < arg_device_max_megabytes) /* Or there is enough collected Neurons for a Partial solution */
  ){
    visiting_next = get_next_neuron(visiting, strict);
    if(visiting.back() == visiting_next)
      add_neuron_into_subset(visiting.back());
    step(visiting, visiting_next);
  }
}

std::uint32_t NeuronRouter::get_next_neuron(std::vector<std::uint32_t>& visiting, bool strict){
  std::uint32_t visiting_next = 0;
  std::uint32_t start_input_index_from = 0;
  std::uint32_t number_of_processed_inputs = 0;
  std::uint32_t start_synapse_iteration_from = 0;
  std::uint32_t expected_neuron_state = 0;

  visiting_next = visiting.back();
  while(/* Checking current Neuron and its inputs */
    (is_neuron_subset_candidate(visiting.back(),iteration))
    &&(number_of_processed_inputs < neuron_number_of_inputs[visiting.back()]) /* Neuron has some unprocessed and not reserved inputs */
    &&(visiting.back() == visiting_next)  /* no children are found to move on to */
  ){
    SynapseIterator<InputSynapseInterval> iter(net.neuron_array(visiting.back()).input_indices());
    expected_neuron_state = *neuron_states[visiting.back()];
    if(is_neuron_in_progress(visiting.back())){ /* If the Neuron is in progess still */
      number_of_processed_inputs = std::min(
        static_cast<std::uint32_t>(*neuron_states[visiting.back()]),
        neuron_number_of_inputs[visiting.back()]
      );
      iter.skim_terminatable([&](InputSynapseInterval input_synapse){
        if((start_input_index_from + input_synapse.interval_size()) < number_of_processed_inputs){
          ++start_synapse_iteration_from; /* Skip this synapse */
          start_input_index_from += input_synapse.interval_size();
          return true; /* start_input_index_from was still smaller, than number_of_processed_inputs, so the synapse can be skipped */
        }else return false; /* Can't skip anymore synapses, as Neuron state implies, current synapse is in progress */
      }); /* Skim through its input synapses */
    }
    number_of_processed_inputs = start_input_index_from;
    std::uint32_t current_backreach;
    iter.iterate_terminatable([&](InputSynapseInterval input_synapse){
      current_backreach = input_synapse.reach_past_loops();
      return true;
    },[&](std::int32_t synapse_input_index){
      if(
        ( /* check if the input is the Neuron is itself, because then it automatically counts as finished */
          (!SynapseIterator<>::is_index_input(synapse_input_index))
          &&(static_cast<std::int32_t>(visiting.back()) == synapse_input_index)
        )||(( /* check for each input of the neuron if they are finished */
          (SynapseIterator<>::is_index_input(synapse_input_index))
          ||(0u < current_backreach) /* Neuron Inputs from the past count as already processed */
          ||(is_neuron_processed(synapse_input_index))
          ||((!strict)&&(is_neuron_reserved(synapse_input_index)))
         )&&(
           (SynapseIterator<>::is_index_input(synapse_input_index))
           ||(0u < current_backreach) /* Neuron Inputs from the past count as already processed */
           ||(are_neuron_feature_groups_finished_for(synapse_input_index))
           /*!Note: the index reference is never called with Inputs ( with negative index values ) because of the || clause */
       ))
      ){
        ++number_of_processed_inputs;
        return true;
      }else if(
        (!SynapseIterator<>::is_index_input(synapse_input_index))
        &&(is_neuron_subset_candidate(synapse_input_index, iteration))
      ){
        visiting_next = synapse_input_index;
        return  false;
      }else return true;
    },start_synapse_iteration_from);
    if( /* Some inputs are still unprocessed */
      (number_of_processed_inputs < neuron_number_of_inputs[visiting.back()])
      &&(visiting_next == visiting.back()) /* There are no next input to iterate to */
    ){
      (void)neuron_states[visiting.back()]->compare_exchange_strong(
        expected_neuron_state,
        neuron_state_next_iteration_value(visiting.back(),iteration)
      ); /* If another thread updated the Neuron status before this one, don't tinker with it! */
    }else{ /* Neuron has unprocessed inputs still, iteration shall continue with one of them */
      (void)neuron_states[visiting.back()]->compare_exchange_strong(
        expected_neuron_state,
        number_of_processed_inputs
      ); /* If another thread updated the Neuron status before this one, don't tinker with it! */
    }
  } /* Checking current Neuron and its inputs */
  return visiting_next;
}

void NeuronRouter::add_neuron_into_subset(std::uint32_t neuron_index){
  std::uint32_t tmp_number = neuron_number_of_inputs[neuron_index];
  double tmp_size = 0;
  std::lock_guard<std::mutex> lock(net_subset_mutex);
  if(
    /* (0 == (rand()%50))&& Uncomment this to test the roboustness of the network creation */
    is_neuron_solvable(neuron_index) /* If Neuron is solvable, and lock is successful */
    &&((neuron_states[neuron_index])->compare_exchange_strong(tmp_number,neuron_state_reserved_value(neuron_index)))
  ){ /* Push it into the Neuron subset */
    for(std::uint32_t subset_neuron_index : net_subset){
      if(subset_neuron_index == neuron_index){
        return;
      }/* If it's already in there, exit this function. */
    } /* Check if the subset already contains @neuron_index */

    net_subset.push_back(neuron_index);
    net_subset_index.push_back(std::numeric_limits<std::uint32_t>::max());

    /* Collect estimated size of Neuron in the @PartialSolution */
    tmp_number = NeuronInfo::get_neuron_estimated_size_bytes(net.neuron_array(neuron_index));
    tmp_size = net_subset_size_bytes; /* Add estimated Neuron Size */
    while(!net_subset_size_bytes.compare_exchange_weak(
      tmp_size,tmp_size + static_cast<double>(tmp_number)/((1024.0) * (1024.0))
    ))tmp_size = net_subset_size_bytes;
  }
}

void NeuronRouter::step(std::vector<std::uint32_t>& visiting, std::uint32_t visiting_next){
  std::uint32_t tmp_index = 0;
  if(visiting_next != visiting.back()){ /* found another Neuron to iterate to because the index values differ (because visiting_next is updated!) */
    visiting.push_back(visiting_next);
  }else if(1 < visiting.size()){ /* haven't found another Neuron to iterate to, try with parent Neuron, if there is any.. */
    visiting.pop_back(); /* --> remove latest Neuron from the queue, go to its parent in the next iteration */
  }
  if(1 == visiting.size()){ /* The Visiting vector is down to it's last element, which is the visit-starting output layer neuron */
    tmp_index = visiting.back();
    if((!is_neuron_in_progress(tmp_index))&&(!is_neuron_subset_candidate(tmp_index, iteration))){
      visiting.back()++; /* If Neuron is processed, reserved or not relevant to the current iteration go to the next one */
    }/*!Note: It is possible to get out of bounds here, it will mean that this thread is finished, and collection ( if needed ) will restart in the next iteration */
    if(
      (is_neuron_processed(tmp_index))&&(tmp_index == output_layer_iterator) /* If the Neuron at @output_layer_iterator is processed */
      &&(static_cast<int>(output_layer_iterator) < (net.neuron_array_size()-1)) /* And it shall remain in bounds of the array */
    ){ /*  step the output_layer_iterator forward! */
      (void)output_layer_iterator.compare_exchange_strong( tmp_index, (tmp_index+1) );
      /*!Note: @output_layer_iterator may have been updated within another thread, but that's okay */
    }
  } /* (1 == visiting.size()) */
}

std::vector<std::uint32_t> NeuronRouter::confirm_first_subset_element_processed(std::uint32_t neuron_index){
  RFASSERT(!collection_running);
  RFASSERT(0 < net_subset.size());
  RFASSERT(neuron_index == net_subset.front());
  std::vector<std::uint32_t> solved_features; /* of index values pointing to tracked_features */
  std::vector<std::uint32_t> retval; /* of index value pointing to network feature groups */

  (neuron_states[neuron_index])->store(neuron_state_processed_value(neuron_index));
  net_subset.pop_front();

  /* Look for the solved features */
  for(std::uint32_t feature_index = 0; feature_index < features_assigned_to_neurons[neuron_index].size(); feature_index++){
    FeatureGroupCache& feature = tracked_features[features_assigned_to_neurons[neuron_index][feature_index]];
    feature.neuron_triggered();
    if(feature.solved()){
      solved_features.push_back(features_assigned_to_neurons[neuron_index][feature_index]);
      retval.push_back(feature.get_index());
    }
  }

  for(std::uint32_t feature_index : solved_features){ /* Removed already solved features from Neuron feature vector cache */
    const FeatureGroup& solved_feature_group = net.neuron_group_features(tracked_features[feature_index].get_index());
    SynapseIterator<>::iterate(solved_feature_group.relevant_neurons(), [&](std::int32_t neuron_index){
      /* find feature in Neuron assigned features */
      std::vector<std::uint32_t>& neuron_features_vecor = features_assigned_to_neurons[neuron_index];
      std::vector<std::uint32_t>::iterator it = std::find(neuron_features_vecor.begin(), neuron_features_vecor.end(), feature_index);
      if(it != neuron_features_vecor.end()){ neuron_features_vecor.erase(it); }
    });
  }
  return retval;
}

bool NeuronRouter::is_neuron_without_dependency(std::uint32_t neuron_index){
  bool ret = true;
  if(!is_neuron_processed(neuron_index)){
    std::deque<std::uint32_t>::iterator neuron_in_subset = std::find(net_subset.begin(), net_subset.end(), neuron_index);
    if(net_subset.end() != neuron_in_subset){ /* The Neuron must be included in the subset if it's not processed already to not have any dependencies */
      /* The Neuron is not processed, but included in the subset. Check its inputs! */
      SynapseIterator<InputSynapseInterval>::iterate_terminatable(net.neuron_array(neuron_index).input_indices(),[&](std::int32_t synapse_input_index){
        if(!is_neuron_processed(synapse_input_index)){ /* If Neuron input is not processed */
          /* then the input must be in front of the Neuron inside the subset */
          for(std::deque<std::uint32_t>::iterator iter = net_subset.begin(); iter != neuron_in_subset; ++iter){
            if(static_cast<std::int32_t>(*iter) == synapse_input_index)
              return true; /* Found the Neuron input before its parent! Input OK, but continue searching. */
          }
          ret = false; /* Could not find Neuron input before the Neuron in the subset */
          return false; /* No need to continue the search, because the Neuron has pending dependencies! */
        }else return true; /* The Neuron input is processed, continue the examination.. */
      });
      return ret;
    }else return false; /* The Neuron is not even in the subset while being unprocessed, it has dependencies. */
  }return true; /* Neuron is already processed, theoritically it shouldn't have any pending dependecies.. */
}

std::vector<std::uint32_t> NeuronRouter::get_dependents_in_subset_of(std::uint32_t neuron_index) const{
  std::vector<std::uint32_t> result;
  if(0 < net_subset.size()){
    /* Find Neuron in subset */
    std::uint32_t subset_index = net_subset.size();
    for(std::uint32_t subset_iterator = 0; subset_iterator < net_subset.size(); ++subset_iterator){
      if(neuron_index == net_subset[subset_iterator]){
        subset_index = subset_iterator;
        result.push_back(neuron_index);
        break;
      }
    }
    if(subset_index < net_subset.size()){ /* Neuron was found in subset! */
      for(std::uint32_t subset_iterator = 0; subset_iterator < net_subset.size(); ++subset_iterator){
        /* go through the subset and omit the Neurons who have this one as a dependency */
        SynapseIterator<InputSynapseInterval>::iterate(
          net.neuron_array(net_subset[subset_iterator]).input_indices(),
          [&](std::int32_t synapse_index){
            if(synapse_index == static_cast<std::int32_t>(neuron_index)){
              result.push_back(net_subset[subset_iterator]);
            }
          }
        );
      }
    } /* else neuron index was not found in the subset! */
  }
  return result;
}

void NeuronRouter::omit_from_subset(std::uint32_t neuron_index){
  if(collection_running)throw std::runtime_error("Unable to omit Neuron because subset colleciton is still ongoing!");
  std::vector<std::uint32_t> to_remove = get_dependents_in_subset_of(neuron_index);
  for(std::uint32_t neuron : to_remove){
    (neuron_states[neuron])->store(0); /* set its state back to 0 */
    for(std::uint32_t subset_iterator = 0; subset_iterator < net_subset.size(); ++subset_iterator){
      if(net_subset[subset_iterator] == neuron){ /* And then erase it from the subset finally */
        net_subset_size_bytes.store(net_subset_size_bytes.load() - NeuronInfo::get_neuron_estimated_size_bytes(net.neuron_array(neuron)));
        net_subset.erase(net_subset.begin() + subset_iterator);
        break;
      }
    }
  }
  for(std::uint32_t neuron : to_remove){
    omit_from_subset(neuron);
  }
}

void NeuronRouter::omit_from_subset(std::uint32_t neuron_index, std::deque<std::uint32_t>& paired_array){
  if(collection_running)throw std::runtime_error("Unable to omit Neuron because subset colleciton is still ongoing!");
  if(get_subset_size() != paired_array.size()) throw std::runtime_error("Subset size doesn't match with the paired array!");
  std::vector<std::uint32_t> to_remove = get_dependents_in_subset_of(neuron_index);
  for(std::uint32_t neuron : to_remove){
    (neuron_states[neuron])->store(0); /* set its state back to 0 */
    for(std::uint32_t subset_iterator = 0; subset_iterator < net_subset.size(); ++subset_iterator){
      if(net_subset[subset_iterator] == neuron){ /* And then erase it from the subset finally */
        net_subset_size_bytes.store(net_subset_size_bytes.load() - NeuronInfo::get_neuron_estimated_size_bytes(net.neuron_array(neuron)));
        net_subset.erase(net_subset.begin() + subset_iterator);
        paired_array.erase(paired_array.begin() + subset_iterator);
        break;
      }
    }
  }
  for(std::uint32_t neuron : to_remove){
    omit_from_subset(neuron, paired_array);
  }
}

std::uint32_t NeuronRouter::neuron_state_reserved_value(std::uint32_t neuron_index) const{
  RFASSERT(neuron_index < neuron_number_of_inputs.size());
  return neuron_number_of_inputs[neuron_index] + 1u;
}

std::uint32_t NeuronRouter::neuron_state_processed_value(std::uint32_t neuron_index) const{
  RFASSERT(neuron_index < neuron_number_of_inputs.size());
  return neuron_number_of_inputs[neuron_index] + 2u;
}

std::int32_t NeuronRouter::neuron_state_iteration_value(std::uint32_t neuron_index) const{
  return (*neuron_states[neuron_index] - neuron_state_processed_value(neuron_index));
}

std::uint32_t NeuronRouter::neuron_iteration_relevance(std::uint32_t neuron_index) const{
  return static_cast<std::uint32_t>(std::max( 0, neuron_state_iteration_value(neuron_index) ));
}

std::int32_t NeuronRouter::neuron_state_next_iteration_value(std::uint32_t neuron_index, std::uint16_t iteration) const{
  return (neuron_state_processed_value(neuron_index) + iteration + 1u);
}

bool NeuronRouter::is_neuron_subset_candidate(std::uint32_t neuron_index, std::uint16_t iteration) const{
  return(
    (neuron_iteration_relevance(neuron_index) <= iteration)
    &&(!is_neuron_processed(neuron_index))
    &&(!is_neuron_reserved(neuron_index))
  );
}

bool NeuronRouter::are_neuron_feature_groups_finished_for(std::uint32_t neuron_index) const{
  RFASSERT(neuron_index < features_assigned_to_neurons.size());
  for(std::uint32_t feature_index = 0; feature_index < features_assigned_to_neurons[neuron_index].size(); feature_index++){
    const FeatureGroupCache& feature_group_cache = tracked_features[features_assigned_to_neurons[neuron_index][feature_index]];
    const FeatureGroup& feature_group = net.neuron_group_features(feature_group_cache.get_index());
    if( (NeuronInfo::is_feature_relevant_to_solution(feature_group.feature()))&&(!feature_group_cache.solved()) )
      return false;
  }
  return true;
}

} /* namespace rafko_net */

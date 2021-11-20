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

#include "rafko_net/models/neuron_info.h"

#include "rafko_net/services/synapse_iterator.h"

namespace rafko_net{

NeuronRouter::NeuronRouter(const RafkoNet& rafko_net) : net(rafko_net){
  output_layer_iterator = (net.neuron_array_size() - net.output_neuron_number()); /* Start to process Ouptut Layer Neurons */
  neuron_states = vector<unique_ptr<atomic<uint32>>>(); /* Every Neuron has 0 child processed at first */
  neuron_number_of_inputs = vector<uint32>(net.neuron_array_size(),0);
  iteration = 1; /* Has to start with 1, otherwise values mix with neuron processed value */
  for(int neuron_iterator = 0; neuron_iterator < net.neuron_array_size(); ++neuron_iterator){
      for(int synapse_iterator = 0;
        synapse_iterator < net.neuron_array(neuron_iterator).input_indices_size();
        ++synapse_iterator
      ) neuron_number_of_inputs[neuron_iterator] += net.neuron_array(neuron_iterator).input_indices(synapse_iterator).interval_size();
      neuron_states.push_back(std::make_unique<atomic<uint32>>());
  } /* Calculating how many children one Neuron has */
  net_subset_size_bytes = double_literal(0.0);
  net_subset = std::deque<uint32>();
  net_subset_index = std::deque<uint32>();
  collection_running = false;
}

void NeuronRouter::collect_subset(uint8 arg_max_solve_threads, sdouble32 arg_device_max_megabytes, bool strict){

  using std::thread;

  collection_running = true;
  vector<thread> processing_threads;
  for(uint8 thread_iterator = 0; thread_iterator < arg_max_solve_threads; thread_iterator++){
    processing_threads.push_back(
      thread(
        &NeuronRouter::collect_subset_thread,
        this, arg_max_solve_threads, arg_device_max_megabytes, thread_iterator, strict
      )
    ); /* Add threads for processing */
  }

  std::for_each(processing_threads.begin(),processing_threads.end(),[](thread& processing_thread){
    if(true == processing_thread.joinable())processing_thread.join();
  });
  if(strict) /* in strict mode, all Neurons are independent, so the order of the queue doesn't matter */
    std::sort(net_subset.begin(),net_subset.end());
  collection_running = false;
  ++iteration;
}

void NeuronRouter::collect_subset_thread(uint8 arg_max_solve_threads, sdouble32 arg_device_max_megabytes, uint8 thread_index, bool strict){
  /**
   * In order of the iteration, the visited neuron indexes. The First Index is always one of the Output Layer Neurons
   */
  vector<uint32> visiting(1,
    (output_layer_iterator + ((net.neuron_array_size()-1-output_layer_iterator)/arg_max_solve_threads)*thread_index)
  ); /* The first Neuron to be visited is decided based on the number of threads, to make sure the threads are as independent as possible */
  uint32 visiting_next = visiting.back();

  while( /* Iterate the Net until every possible Neuron is collected into an independent subset of it */
    (net.neuron_array_size() > static_cast<int>(visiting.back())) /* The currently visiting Neuron is inside bounds of the net */
    &&(static_cast<int>(output_layer_iterator) < net.neuron_array_size()) /* Until the whole output layer is processed */
    &&(net_subset_size_bytes/* Bytes *// double_literal(1024.0) /* KB *// double_literal(1024.0) /* MB */ < arg_device_max_megabytes) /* Or there is enough collected Neurons for a Partial solution */
  ){
    visiting_next = get_next_neuron(visiting, strict);
    if(visiting.back() == visiting_next)
      add_neuron_into_subset(visiting.back());
    step(visiting, visiting_next);
  }
}

uint32 NeuronRouter::get_next_neuron(vector<uint32>& visiting, bool strict){
  uint32 visiting_next = 0;
  uint32 start_input_index_from = 0;
  uint32 number_of_processed_inputs = 0;
  uint32 start_synapse_iteration_from = 0;
  uint32 expected_neuron_state = 0;

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
        static_cast<uint32>(*neuron_states[visiting.back()]),
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
    iter.iterate_terminatable([&](InputSynapseInterval input_synapse, sint32 synapse_input_index){
      if(
        (SynapseIterator<>::is_index_input(synapse_input_index))
        ||(0 < input_synapse.reach_past_loops()) /* Inputs from the past count as already processed */
        ||(is_neuron_processed(synapse_input_index))
        ||((!strict)&&(is_neuron_reserved(synapse_input_index)))
        /*!Note: In non-strict mode usually the whole of the net is collected into the subset in order,
         * which might be undesirable compared to the Neurons being collected into smaller non-dependent subsets.
         **/
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

void NeuronRouter::add_neuron_into_subset(uint32 neuron_index){
  uint32 tmp_number = neuron_number_of_inputs[neuron_index];
  sdouble32 tmp_size = 0;
  std::lock_guard<std::mutex> lock(net_subset_mutex);
  if(
    /* (0 == (rand()%50))&& Uncomment this to test the roboustness of the network creation */
    is_neuron_solvable(neuron_index) /* If Neuron is solvable, and lock is successful */
    &&((neuron_states[neuron_index])->compare_exchange_strong(tmp_number,neuron_state_reserved_value(neuron_index)))
  ){ /* Push it into the Neuron subset */
    for(uint32 subset_neuron_index : net_subset){
      if(subset_neuron_index == neuron_index){
        return;
      }/* If it's already in there, exit this function. */
    } /* Check if the subset already contains @neuron_index */

    net_subset.push_back(neuron_index);
    net_subset_index.push_back(std::numeric_limits<uint32>::max());

    /* Collect estimated size of Neuron in the @PartialSolution */
    tmp_number = NeuronInfo::get_neuron_estimated_size_bytes(net.neuron_array(neuron_index));
    tmp_size = net_subset_size_bytes; /* Add estimated Neuron Size */
    while(!net_subset_size_bytes.compare_exchange_weak(
      tmp_size,tmp_size + static_cast<sdouble32>(tmp_number)/(double_literal(1024.0) * double_literal(1024.0))
    ))tmp_size = net_subset_size_bytes;
  }
}

void NeuronRouter::step(vector<uint32>& visiting, uint32 visiting_next){
  uint32 tmp_index = 0;
  if(visiting_next != visiting.back()){ /* found another Neuron to iterate to because the index values differ (because visiting_next is updated!) */
    visiting.push_back(visiting_next);
  }else if(1 < visiting.size()){ /* haven't found another Neuron to iterate to, try with parent Neuron, if there is any */
    visiting.pop_back(); /* remove latest Neuron from the queue, go to its parent in the next iteration */
  }
  if(1 == visiting.size()){ /* The Visiting vector is down to it's last element, which is the visit-starting output layer neuron */
    tmp_index = visiting.back();
    if((!is_neuron_in_progress(tmp_index))&&(!is_neuron_subset_candidate(tmp_index, iteration))){
      visiting.back()++; /* If Neuron is processed, reserved or not relevant to the current iteration go to the next one */
    }/*!Note: It is possible to get out of bounds here, it will mean that this thread is finished, and collection ( if needed ) will restart in the next iteration */
    if(
      (is_neuron_processed(tmp_index))
      &&(tmp_index == output_layer_iterator) /* If the Neuron at @output_layer_iterator is processed */
      &&(static_cast<int>(output_layer_iterator) < (net.neuron_array_size()-1)) /* And it shall remain in bounds of the array */
    ){ /*  step the output_layer_iterator forward! */
      (void)output_layer_iterator.compare_exchange_strong( tmp_index, (tmp_index+1) );
      /*!Note: @output_layer_iterator may have been updated within another thread, but that's okay */
    }
  } /* (1 == visiting.size()) */
}

bool NeuronRouter::is_neuron_without_dependency(uint32 neuron_index){
  bool ret = true;
  if(!is_neuron_processed(neuron_index)){
    deque<uint32>::iterator neuron_in_subset = std::find(net_subset.begin(), net_subset.end(), neuron_index);
    if(net_subset.end() != neuron_in_subset){ /* The Neuron must be included in the subset if it's not processed already to not have any dependencies */
      /* The Neuron is not processed, but included in the subset. Check its inputs! */
      SynapseIterator<InputSynapseInterval>::iterate_terminatable(net.neuron_array(neuron_index).input_indices(),
      [&](InputSynapseInterval input_synapse, sint32 synapse_input_index){
        parameter_not_used(input_synapse);
        if(!is_neuron_processed(synapse_input_index)){ /* If Neuron input is not processed */
          /* then the input must be in front of the Neuron inside the subset */
          for(deque<uint32>::iterator iter = net_subset.begin(); iter != neuron_in_subset; ++iter){
            if(static_cast<sint32>(*iter) == synapse_input_index)
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

vector<uint32> NeuronRouter::get_dependents_in_subset_of(uint32 neuron_index){
  vector<uint32> result;
  if(0 < net_subset.size()){
    /* Find Neuron in subset */
    uint32 subset_index = net_subset.size();
    for(uint32 subset_iterator = 0; subset_iterator < net_subset.size(); ++subset_iterator){
      if(neuron_index == net_subset[subset_iterator]){
        subset_index = subset_iterator;
        result.push_back(neuron_index);
        break;
      }
    }
    if(subset_index < net_subset.size()){ /* Neuron was found in subset! */
      for(uint32 subset_iterator = 0; subset_iterator < net_subset.size(); ++subset_iterator){
        /* go through the subset and omit the Neurons who have this one as a dependency */
        SynapseIterator<InputSynapseInterval>::iterate(
          net.neuron_array(net_subset[subset_iterator]).input_indices(),
          [&](InputSynapseInterval input_synapse, sint32 synapse_index){
            parameter_not_used(input_synapse);
            if(synapse_index == static_cast<sint32>(neuron_index)){
              result.push_back(net_subset[subset_iterator]);
            }
          }
        );
      }
    } /* else neuron index was not found in the subset! */
  }
  return result;
}

void NeuronRouter::omit_from_subset(uint32 neuron_index){
  if(collection_running)throw std::runtime_error("Unable to omit Neuron because subset colleciton is still ongoing!");
  vector<uint32> to_remove = get_dependents_in_subset_of(neuron_index);
  for(uint32 neuron : to_remove){
    (neuron_states[neuron])->store(0); /* set its state back to 0 */
    for(uint32 subset_iterator = 0; subset_iterator < net_subset.size(); ++subset_iterator){
      if(net_subset[subset_iterator] == neuron){ /* And then erase it from the subset finally */
        net_subset_size_bytes.store(net_subset_size_bytes.load() - NeuronInfo::get_neuron_estimated_size_bytes(net.neuron_array(neuron)));
        net_subset.erase(net_subset.begin() + subset_iterator);
        break;
      }
    }
  }
  for(uint32 neuron : to_remove){
    omit_from_subset(neuron);
  }
}

void NeuronRouter::omit_from_subset(uint32 neuron_index, deque<uint32>& paired_array){
  if(collection_running)throw std::runtime_error("Unable to omit Neuron because subset colleciton is still ongoing!");
  if(get_subset_size() != paired_array.size()) throw std::runtime_error("Subset size doesn't match with the paired array!");
  vector<uint32> to_remove = get_dependents_in_subset_of(neuron_index);
  for(uint32 neuron : to_remove){
    (neuron_states[neuron])->store(0); /* set its state back to 0 */
    for(uint32 subset_iterator = 0; subset_iterator < net_subset.size(); ++subset_iterator){
      if(net_subset[subset_iterator] == neuron){ /* And then erase it from the subset finally */
        net_subset_size_bytes.store(net_subset_size_bytes.load() - NeuronInfo::get_neuron_estimated_size_bytes(net.neuron_array(neuron)));
        net_subset.erase(net_subset.begin() + subset_iterator);
        paired_array.erase(paired_array.begin() + subset_iterator);
        break;
      }
    }
  }
  for(uint32 neuron : to_remove){
    omit_from_subset(neuron, paired_array);
  }
}

} /* namespace rafko_net */

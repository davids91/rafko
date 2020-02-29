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

#include "services/backpropagation_queue_wrapper.h"
#include "services/neuron_router.h"

#include <atomic>
#include <thread>
#include <deque>

namespace sparse_net_library{

Backpropagation_queue_wrapper::Backpropagation_queue_wrapper(SparseNet& net, Service_context context){

  using std::vector;
  using std::deque;
  using std::atomic;
  using std::thread;

  deque<vector<uint32>> neuron_queue = deque<vector<uint32>>(1,vector<uint32>(0));
  vector<thread> sort_threads = vector<thread>();
  Neuron_router neuron_router(net);
  uint32 neuron_index;
  uint32 neuron_depth = 0;
  uint32 neurons_done = 0;
  uint32 neuron_depth_done = 0;
  gradient_step = Backpropagation_queue();

  while(net.neuron_array_size() > static_cast<int>(neurons_done)){
    /* Collect a strict subset from the net */
    neuron_router.collect_subset(context.get_max_solve_threads(),context.get_device_max_megabytes(),true);
    while(neuron_router.get_first_neuron_index_from_subset(neuron_index)){
      neuron_queue.back().push_back(neuron_index);
      ++neurons_done;
      neuron_router.confirm_first_subset_element_processed(neuron_index);
    }

    if(0 < neuron_queue.back().size()){ /* Add them into the queue */
      sort_threads.push_back(thread([&](){ /* Sort the thread in ascending for synapse compression */
        std::sort(neuron_queue[neuron_depth].begin(),neuron_queue[neuron_depth].begin());
        ++neuron_depth_done;
      }));
      ++neuron_depth;
      neuron_queue.push_back(vector<uint32>());
    }
  } /* while(net.neuron_array_size() > static_cast<int>(neurons_done)) */

  while(0 < sort_threads.size()){
    sort_threads.back().join();
    sort_threads.pop_back();
  }

  /* Push queue array into gradient step */
  uint32 previous_added_index = -1;
  uint32 number_of_neurons_in_depth;
  Synapse_interval tmp_interval = Synapse_interval();
  for(auto depth_iterator = neuron_queue.rbegin(); depth_iterator != neuron_queue.rend(); ++depth_iterator){
    number_of_neurons_in_depth = 0;
    for(uint32 neuron_index : *depth_iterator){
      if(
        (0 == gradient_step.neuron_synapses_size())
        ||((neuron_index-1) != previous_added_index)
      ){ /* Open up a new synapse */
        tmp_interval.set_starts(neuron_index);
        tmp_interval.set_interval_size(1);
        *gradient_step.add_neuron_synapses() = tmp_interval;
      }else{ /* Extend the latest synapse */
        gradient_step.mutable_neuron_synapses(gradient_step.neuron_synapses_size()-1)->set_interval_size(
        gradient_step.neuron_synapses(gradient_step.neuron_synapses_size()-1).interval_size() + 1);
      }
      ++number_of_neurons_in_depth;
      previous_added_index = neuron_index;
    } /* for(uint32 neuron_index : *depth_iterator){ */
    if(0 < number_of_neurons_in_depth)gradient_step.add_cols(number_of_neurons_in_depth);
  }
}

} /* namespace sparse_net_library */

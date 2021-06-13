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

#ifndef NEURON_ROUTER_H
#define NEURON_ROUTER_H

#include "rafko_global.h"

#include <deque>
#include <mutex>
#include <atomic>
#include <functional>
#include <stdexcept>

#include "gen/sparse_net.pb.h"

namespace sparse_net_library {

using std::unique_ptr;
using std::vector;
using std::deque;
using std::atomic;
using std::mutex;

/**
 * @brief      This class describes a neuron router which iterates through the given @SparseNet,
               collecting a subset of Neurons from the thread, all of whom are able to be solved without
               waiting for any other Neurons. The subset is being collected based on the input relations
               between the Neurons. The Neurons at the beginning of the net only take in input data,
               so they already have their inputs ready. Any other Neurons build upon that, with each Iteration
               some additional @Neuron nodes are collected into a subset. That subset is later to be used by
               the @Solution_builder to compile @Partial_solutions.
               If a Neuron is solvable, its state is being set to "reserved", and collected into the subset.
               After an iteration the state update from the subset needs to be handled by whoever has access to
               the Neuron indexes inside.
 */
class Neuron_router{
public:
  Neuron_router(const SparseNet& sparse_net);

  uint32 operator[](int index){
    return get_neuron_index_from_subset(index);
  }

  /**
   * @brief      Collects some Neurons into a solvable subset of the net
   *
   * @param[in]  arg_max_solve_threads     The argument maximum solve threads
   * @param[in]  arg_device_max_megabytes  The argument device maximum megabytes
   * @param[in]  strict                    In strict mode, reserved Neurons do not count as finished
   */
  void collect_subset(uint8 arg_max_solve_threads, sdouble32 arg_device_max_megabytes, bool strict = true);

  /**
   * @brief      Reads an index from the recently collected subset. Shall only be used
   *             when a collection is not actually running; It's not thread-safe!
   *
   * @param[in]  subset_index  The subset index
   *
   * @return     The neuron index from subset.
   */
  uint32 get_neuron_index_from_subset(uint32 subset_index) const{
    if((!collection_running)&&(0 < net_subset.size())){
      return net_subset[subset_index];
    }else throw std::runtime_error("Invalid usage or Index out of bounds!");
  }

  /**
   * @brief      Gets the first neuron index from the recently collected subset, if there is any.
   *
   * @param      put_it_here  The reference to load the index into
   *
   * @return     Operation success
   */
  bool get_first_neuron_index_from_subset(uint32& put_it_here) const{
    if((!collection_running)&&(0 < net_subset.size())){
      put_it_here = net_subset.front();
      return true;
    }else return false;
  }

  /**
   * @brief      If the index in the arguments matches the first index in the subset,
   *             removes the index from it; Sets its neuron state to processed.
   *             This validation mechanism is to ensure that the user
   *             of the interface knows what index it is removing.
   *
   * @param[in]  neuron_index  The neuron index to compare against
   *
   * @return     Operation success
   */
  bool confirm_first_subset_element_processed(uint32 neuron_index){
    if((!collection_running)&&(0 < net_subset.size())&&(neuron_index == net_subset.front())){
      (neuron_states[neuron_index])->store(neuron_state_processed_value(neuron_index));
      net_subset.pop_front();
      return true;
    }else return false;
  }

  /**
   * @brief      If the index in the arguments matches the first index in the subset,
   *             removes the index from it; Sets its neuron state to be in progress.
   *             This validation mechanism is to ensure that the user
   *             of the interface knows what index it is removing.
   *
   * @param[in]  neuron_index  The neuron index to compare against
   *
   * @return     Operation success
   */
  bool confirm_first_subset_element_ommitted(uint32 neuron_index){
    if((0 < net_subset.size())&&(neuron_index == net_subset.front())){
      omit_from_subset(neuron_index);
      return true;
    } else return false;
  }

  /**
   * @brief      Same functionality as the other function with the same name, except it
   *             accepts a helper array which is intended to store information related to
   *             the subset, so by modifying the subset, it needs to be modified as well.
   *             Sizes must always equal!
   *
   * @param[in]  neuron_index  The neuron index to be removed
   * @param      paired_array  The helper array to be modified in accordance with the subset
   *
   * @return     { description_of_the_return_value }
   */
  bool confirm_first_subset_element_ommitted(uint32 neuron_index, deque<uint32>& paired_array){
    if(
      (0 < net_subset.size())&&(neuron_index == net_subset.front())
      &&(net_subset.size() == paired_array.size())
    ){
      omit_from_subset(neuron_index, paired_array);
      return true;
    } else return false;
  }

  /**
   * @brief      Resets the neurons in the subset for all but the ones provided in the
   *             argument. The list has to match the subset exactly, or the function
   *             throws an exception.
   *
   * @param[in]  the_front  The front
   */
  void reset_all_except(vector<uint32> the_front){
    uint32 front_index = 0;
    for(uint32 subset_index : net_subset){
      if(the_front.size() == front_index)
        break; /* Only go through the front, for checking. The remaining Neurons shall be resetted */
      if(subset_index != the_front[front_index]){ /* The indices have to match */
        throw std::runtime_error("Subset mismatch!");
      }else ++front_index;
    }
    /* This point is not reachable if there is a mismatch */
    net_subset.resize(the_front.size()); /* This also means, that the front is an exact first part of the subset */
  }

  /**
   * @brief      Gets the number of elements in the subset
   *
   * @return     The subset size.
   */
  uint32 get_subset_size() const{
    return net_subset.size();
  }

  /**
   * @brief      Gets a non-modifyable reference to the currently collected subset of Neuron indices.
   *
   * @return     The subset.
   */
  const deque<uint32>& get_subset() const{
    return net_subset;
  }

  /**
   * @brief      Clears the subset and sets the neuron states of the items in it to be in progress.
   */
  void reset_remaining_subset(void){
    while(0 < net_subset.size())
      confirm_first_subset_element_ommitted(net_subset.front());
  }

  /**
   * @brief      Gives back Iteration state
   *
   * @return     true if the crrent iteration of the net is finished and resulted with a subset of it
   */
  bool finished() const{
    return (
      (static_cast<int>(output_layer_iterator) == (net.neuron_array_size()-1))
      &&(is_neuron_processed(output_layer_iterator))
    );
  }

  /**
   * @brief      Determines whether the specified neuron is without any pending dependencies.
   *             A Neuron is without dependency if it's every child is either already processed,
   *             or inside the currently collected subset, in front of the neuron.
   *
   * @param[in]  neuron_index  The index of the Neuron to examine
   *
   * @return     True if the specified neuron index is without pending dependency, False otherwise.
   */
  bool is_neuron_without_dependency(uint32 neuron_index);

  bool is_neuron_in_progress(uint32 neuron_index) const{
    return (neuron_number_of_inputs[neuron_index] > *neuron_states[neuron_index]);
  }
  bool is_neuron_reserved(uint32 neuron_index) const{
    return (neuron_state_reserved_value(neuron_index) == *neuron_states[neuron_index]);
  }
  bool is_neuron_solvable(uint32 neuron_index) const{
    return (neuron_number_of_inputs[neuron_index] == *neuron_states[neuron_index]);
  }
  bool is_neuron_processed(uint32 neuron_index) const{
    return (neuron_state_processed_value(neuron_index) == *neuron_states[neuron_index]);
  }
private:
  const SparseNet& net;
  bool collection_running = false;

  /**
   * Number of already processed output layer Neurons
   */
  atomic<uint32> output_layer_iterator;

  /**
   * For each @Neuron in @SparseNet stores the processed state. Values:
   *  - Number of processed children ( storing raw children number without synapse information )
   *  - Number of processed children + 1 in case the Neuron is reserved
   *  - Number of processed children + 2 in case the Neuron is processed
   */
  vector<unique_ptr<atomic<uint32>>> neuron_states;

  /**
   * Number of inputs a Neuron has, based on the input index synapse sizes
   */
  vector<uint32> neuron_number_of_inputs;

  /**
   * A subset of the net representing independent solutions
   */
  mutex net_subset_mutex;
  std::atomic<sdouble32> net_subset_size_bytes; /* The size of the currently partial solution to be built in bytes */
  deque<uint32> net_subset_index;
  deque<uint32> net_subset;

  /**
   * The number of times the algorithm ran to look for Neuron candidates, it is used to decide relevance to the currently finished subset.
   */
  uint16 iteration = 1; /* Has to start with 1, otherwise values mix with neuron processed value */

  /**
   * @brief      Called form inside @collect_subset; A thread to handle @collect_subset
   *
   * @param[in]  arg_max_solve_threads     The argument maximum solve threads
   * @param[in]  arg_device_max_megabytes  The argument device maximum megabytes
   * @param[in]  thread_index              The thread index
   */
  void collect_subset_thread(uint8 arg_max_solve_threads, sdouble32 arg_device_max_megabytes, uint8 thread_index, bool strict);

  /**
   * @brief      Called form inside @collect_subset_thread; Checking the current Neuron and its input states
   *             updates its state accordingly
   *
   * @param      net            The Sparse Net to be used
   * @param      visiting       A Vector containing the currently visiting Neuron along with the path leading to it
   * @return     The next neuron to move the iteration to
   */
  uint32 get_next_neuron(vector<uint32>& visiting, bool strict);

  /**
   * @brief      Called form inside @collect_subset_thread; Adds a neuron into subset and updates relevant build states
   *
   * @param      net            The Sparse Net to be used
   * @param      visiting       A Vector containing the currently visiting Neuron along with the path leading to it
   * @param      visiting_next  The Next Neuron Candidate, which might be the same as the latest visit ( that means no candidates found to move to)
   */
  void add_neuron_into_subset(uint32 neuron_index);

  /**
   * @brief      Removes a Neuron and its dependants from the subset.
   *
   * @param[in]  neuron_index  The neuron index to remove from the subset
   */
  void omit_from_subset(uint32 neuron_index);

  /**
   * @brief      Removes the Neuron and dependants from the subset, including a paired array
   *             of the same size.
   *
   * @param[in]  neuron_index  The neuron index
   * @param      paired_array  The paired array
   */
  void omit_from_subset(uint32 neuron_index, deque<uint32>& paired_array);

  /**
   * @brief      Gets the elements in the current subset depending on the given Neuron index
   *             Including itself.
   *
   * @param[in]  neuron_index  The neuron index
   *
   * @return     A list of the neuron indices inside the subset depending on this one.
   */
  vector<uint32> get_dependents_in_subset_of(uint32 neuron_index);

  /**
   * @brief      Decides the next Neuron to iterate to and increases the output layer iterator if needed
   *
   * @param      visiting       The visiting
   * @param      visiting_next  The visiting next
   */
  void step(vector<uint32>& visiting, uint32 visiting_next);

  /**
   * @brief       functions to help build partial solutions
   *
   * @param[in]  neuron_index  The neuron index inside @neuron_number_of_inputs and @neuron_states
   *
   * @return     Information depending on the function
   */
  uint32 neuron_state_reserved_value(uint32 neuron_index) const{
    return neuron_number_of_inputs[neuron_index] + 1u;
  }
  uint32 neuron_state_processed_value(uint32 neuron_index) const{
    return neuron_number_of_inputs[neuron_index] + 2u;
  }
  sint32 neuron_state_iteration_value(uint32 neuron_index) const{
    return (*neuron_states[neuron_index] - neuron_state_processed_value(neuron_index));
  }
  uint32 neuron_iteration_relevance(uint32 neuron_index) const{
    return static_cast<uint32>(std::max( 0, neuron_state_iteration_value(neuron_index) ));
  }
  sint32 neuron_state_next_iteration_value(uint32 neuron_index, uint16 iteration) const{
    return (neuron_state_processed_value(neuron_index) + iteration + 1u);
  }
   bool is_neuron_subset_candidate(uint32 neuron_index, uint16 iteration) const{
    return(
      (neuron_iteration_relevance(neuron_index) <= iteration)
      &&(!is_neuron_processed(neuron_index))
      &&(!is_neuron_reserved(neuron_index))
    );
  }
};

} /* namespace sparse_net_library */
#endif /* NEURON_ROUTER_H */

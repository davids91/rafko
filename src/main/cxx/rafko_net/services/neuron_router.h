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
#include <vector>
#include <mutex>
#include <atomic>
#include <stdexcept>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_net/services/feature_group_cache.h"

namespace rafko_net {

/**
 * @brief      This class describes a neuron router which iterates through the given @RafkoNet,
 *             collecting a subset of Neurons from the thread, all of who are able to be solved without
 *             waiting for any other Neurons. The subset is being collected based on the input connections
 *             between the Neurons. The Neurons at the beginning of the net only take in input data,
 *             so they already have their inputs ready. Any other Neurons build upon that, with each Iteration
 *             some additional @Neuron nodes are collected into a subset.
 *             If a Neuron is solvable, its state is being set to "reserved", and collected into the subset.
 *             After an iteration the state update from the subset needs to be handled by whoever has access to
 *             the Neuron indexes inside.
 *             In strict mode reserved Neurons do not count as finished, which means Neurons whose inputs are
 *             reserved ( i.e. collected into the subset but not yet processed ) are not collected into the subset.
 *             Non-strict mode enables to collect Neurons into the current subset even if its dependencies are reserved,
 *             so usually the whole of the net is collected into the subset in order in this mode. This might be undesirable
 *             in bigger nets, where the Neurons aimed to be in smaller non-dependent subsets. The subset collected in this mode
 *             is order sensitive, meaning a Neuron in the subset might depend on a different Neuron in the same subset before it,
 *             whereas in strict mode all Neurons are independent, so the order of the queue doesn't matter.
 */
class RAFKO_FULL_EXPORT NeuronRouter{
public:
  NeuronRouter(const RafkoNet& rafko_net);

  std::uint32_t operator[](int index){
    return get_neuron_index_from_subset(index);
  }

  /**
   * @brief      Collects some Neurons into a solvable subset of the net
   *
   * @param[in]  arg_max_solve_threads     The argument maximum solve threads
   * @param[in]  arg_device_max_megabytes  The argument device maximum megabytes
   * @param[in]  strict                    In strict mode, reserved Neurons do not count as finished
   */
  void collect_subset(std::uint8_t arg_max_solve_threads, double arg_device_max_megabytes, bool strict = true);

  /**
   * @brief      Reads an index from the recently collected subset. Shall only be used
   *             when a collection is not actually running; It's not thread-safe!
   *
   * @param[in]  subset_index  The subset index
   *
   * @return     The neuron index from subset.
   */
  std::uint32_t get_neuron_index_from_subset(std::uint32_t subset_index) const{
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
  bool get_first_neuron_index_from_subset(std::uint32_t& put_it_here) const{
    if((!collection_running)&&(0 < net_subset.size())){
      put_it_here = net_subset.front();
      return true;
    } else return false;
  }

  /**
   * @brief      If the index in the arguments matches the first index in the subset,
   *             removes the index from it; Sets its neuron state to processed.
   *             This validation mechanism is to ensure that the user
   *             of the interface knows what index it is removing.
   *
   * @param[in]  neuron_index  The neuron index to compare against
   *
   * @return     List of @neuron_group_features indexes inside the @RafkoNet solved by processing this Neuron
   */
  std::vector<std::uint32_t> confirm_first_subset_element_processed(std::uint32_t neuron_index);

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
  bool confirm_first_subset_element_ommitted(std::uint32_t neuron_index){
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
  bool confirm_first_subset_element_ommitted(std::uint32_t neuron_index, std::deque<std::uint32_t>& paired_array){
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
   * @param[in]  the_front  The front of the subset
   */
  void reset_all_except(std::vector<std::uint32_t> the_front){
    std::uint32_t front_index = 0;
    for(std::uint32_t subset_index : net_subset){
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
  std::uint32_t get_subset_size() const{
    return net_subset.size();
  }

  std::uint32_t get_subset_size_bytes() const{
    return net_subset_size_bytes;
  }

  double get_subset_size_megabytes() const{
    return( static_cast<double>(net_subset_size_bytes) / ((1024.0) * (1024.0)) );
  }

  /**
   * @brief      Gets a non-modifyable reference to the currently collected subset of Neuron indices.
   *
   * @return     The subset.
   */
  constexpr const std::deque<std::uint32_t>& get_subset() const{
    return net_subset;
  }

  /**
   * @brief      Clears the subset and sets the neuron states of the items in it to be in progress.
   */
  void reset_remaining_subset(){
    while(0 < net_subset.size())
      confirm_first_subset_element_ommitted(net_subset.front());
    net_subset_size_bytes.store(0);
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
  bool is_neuron_without_dependency(std::uint32_t neuron_index);

  bool is_neuron_in_progress(std::uint32_t neuron_index) const{
    return (neuron_number_of_inputs[neuron_index] > *neuron_states[neuron_index]);
  }
  bool is_neuron_reserved(std::uint32_t neuron_index) const{
    return (neuron_state_reserved_value(neuron_index) == *neuron_states[neuron_index]);
  }
  bool is_neuron_solvable(std::uint32_t neuron_index) const{
    return (neuron_number_of_inputs[neuron_index] == *neuron_states[neuron_index]);
  }
  bool is_neuron_processed(std::uint32_t neuron_index) const{
    return (neuron_state_processed_value(neuron_index) == *neuron_states[neuron_index]);
  }
private:
  const RafkoNet& net;
  bool collection_running = false;

  /**
   * helper variables representing the relevant features the router needs to consider
   */
  std::vector<FeatureGroupCache> tracked_features;

  /**
   * Number of already processed output layer Neurons
   */
  std::atomic<std::uint32_t> output_layer_iterator;

  /**
   * For each @Neuron in @RafkoNet stores the processed state. Values:
   *  - Number of processed children ( storing raw children number without synapse information )
   *  - Number of processed children + 1 in case the Neuron is reserved
   *  - Number of processed children + 2 in case the Neuron is processed
   */
  std::vector<std::unique_ptr<std::atomic<std::uint32_t>>> neuron_states;

  /**
   * Number of inputs a Neuron has, based on the input index synapse sizes
   */
  std::vector<std::uint32_t> neuron_number_of_inputs;

  /**
   * A vector of index values which points to an element inside of the the tracked feature array
   */
  std::vector<std::vector<std::uint32_t>> features_assigned_to_neurons;

  /**
   * A subset of the net representing independent solutions
   */
  std::mutex net_subset_mutex;
  std::atomic<double> net_subset_size_bytes = (0.0); /* The size of the currently partial solution to be built in bytes */
  std::deque<std::uint32_t> net_subset_index;
  std::deque<std::uint32_t> net_subset;

  /**
   * The number of times the algorithm ran to look for Neuron candidates, it is used to decide relevance to the currently finished subset.
   */
  std::uint16_t iteration = 1; /* Has to start with 1, otherwise values mix with neuron processed value */

  /**
   * @brief      Called form inside @collect_subset; A thread to handle @collect_subset
   *
   * @param[in]  arg_max_solve_threads     The argument maximum solve threads
   * @param[in]  arg_device_max_megabytes  The argument device maximum megabytes
   * @param[in]  thread_index              The thread index
   */
  void collect_subset_thread(std::uint8_t arg_max_solve_threads, double arg_device_max_megabytes, std::uint8_t thread_index, bool strict);

  /**
   * @brief      Called form inside @collect_subset_thread; Checking the current Neuron and its input states
   *             updates its state accordingly
   *
   * @param      net            The Sparse Net to be used
   * @param      visiting       A Vector containing the currently visiting Neuron along with the path leading to it
   * @return     The next neuron to move the iteration to
   */
  std::uint32_t get_next_neuron(std::vector<std::uint32_t>& visiting, bool strict);

  /**
   * @brief      Called form inside @collect_subset_thread; Adds a neuron into subset and updates relevant build states
   *
   * @param      net            The Sparse Net to be used
   * @param      visiting       A Vector containing the currently visiting Neuron along with the path leading to it
   * @param      visiting_next  The Next Neuron Candidate, which might be the same as the latest visit ( that means no candidates found to move to)
   */
  void add_neuron_into_subset(std::uint32_t neuron_index);

  /**
   * @brief      Removes a Neuron and its dependants from the subset.
   *
   * @param[in]  neuron_index  The neuron index to remove from the subset
   */
  void omit_from_subset(std::uint32_t neuron_index);

  /**
   * @brief      Removes the Neuron and dependants from the subset, including a paired array
   *             of the same size.
   *
   * @param[in]  neuron_index  The neuron index
   * @param      paired_array  The paired array
   */
  void omit_from_subset(std::uint32_t neuron_index, std::deque<std::uint32_t>& paired_array);

  /**
   * @brief      Gets the elements in the current subset depending on the given Neuron index
   *             Including itself.
   *
   * @param[in]  neuron_index  The neuron index
   *
   * @return     A list of the neuron indices inside the subset depending on this one.
   */
  std::vector<std::uint32_t> get_dependents_in_subset_of(std::uint32_t neuron_index) const;

  /**
   * @brief      Decides the next Neuron to iterate to and increases the output layer iterator if needed
   *
   * @param      visiting       The visiting
   * @param      visiting_next  The visiting next
   */
  void step(std::vector<std::uint32_t>& visiting, std::uint32_t visiting_next);

  /**
   * @brief      function to support graph traversal of the Neural network
   *
   * @param[in]  neuron_index  The neuron index inside @neuron_number_of_inputs and @neuron_states
   *
   * @return     Provides the value inside inside @neuron_states in case the @Neuron is reserved
   */
  std::uint32_t neuron_state_reserved_value(std::uint32_t neuron_index) const;

  /**
   * @brief      function to support graph traversal of the @RafkoNet
   *
   * @param[in]  neuron_index  The neuron index inside @neuron_number_of_inputs and @neuron_states
   *
   * @return     Provides the value inside inside @neuron_states in case the @Neuron is already processed
   */
  std::uint32_t neuron_state_processed_value(std::uint32_t neuron_index) const;

  /**
   * @brief      function to support graph traversal of the @RafkoNet
   *
   * @param[in]  neuron_index  The neuron index inside @neuron_number_of_inputs and @neuron_states
   *
   * @return     Returns with a the value to be stored inside @neuron_states in case the Neuron is relevant to the current iteration
   */
  std::int32_t neuron_state_iteration_value(std::uint32_t neuron_index) const;

  /**
   * @brief      function to support graph traversal of the @RafkoNet
   *
   * @param[in]  neuron_index  The neuron index inside @neuron_number_of_inputs @features_assigned_to_neurons and @neuron_states
   *
   * @return     Returns with a helper value for the neuron which helps decide if it relevant to the current iteration or only the next one
   */
  std::uint32_t neuron_iteration_relevance(std::uint32_t neuron_index) const;

  /**
   * @brief      function to support graph traversal of the @RafkoNet
   *
   * @param[in]  neuron_index  The neuron index inside @neuron_number_of_inputs @features_assigned_to_neurons and @neuron_states
   *
   * @return     Returns with a the value to be stored inside @neuron_states in case the Neuron is relevant to the nex iteration
   */
  std::int32_t neuron_state_next_iteration_value(std::uint32_t neuron_index, std::uint16_t iteration) const;

  /**
   * @brief       function to support graph traversal of the @RafkoNet
   *
   * @param[in]  neuron_index  The neuron index inside @neuron_number_of_inputs @features_assigned_to_neurons and @neuron_states
   *
   * @return     true if neuron is eligible to become part of the subset currently under construction
   */
  bool is_neuron_subset_candidate(std::uint32_t neuron_index, std::uint16_t iteration) const;

  /**
   * @brief      function to support graph traversal of the @RafkoNet
   *
   * @param[in]  neuron_index  The neuron index inside @neuron_number_of_inputs @features_assigned_to_neurons and @neuron_states
   *
   * @return     all solution relevant feature groups assigned for the Neuron is now finished
   */
  bool are_neuron_feature_groups_finished_for(std::uint32_t neuron_index) const;
};

} /* namespace rafko_net */
#endif /* NEURON_ROUTER_H */

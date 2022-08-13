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

#ifndef RAFKO_WEIGHT_ADAPTER_H
#define RAFKO_WEIGHT_ADAPTER_H

#include "rafko_global.hpp"

#include <vector>
#include <utility>
#include <unordered_map>
#include <mutex>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/solution.pb.h"
#include "rafko_utilities/services/thread_group.hpp"
#include "rafko_mainframe/models/rafko_settings.hpp"

namespace rafko_gym {

/**
 * @brief      Base implementation for updating weights for netowrks based on weight gradients
 */
using PartialWeightPairs = const std::vector<std::pair<std::uint32_t,std::uint32_t>>;
class RAFKO_FULL_EXPORT RafkoWeightAdapter{
public:
  RafkoWeightAdapter(const rafko_net::RafkoNet& rafko_net, rafko_net::Solution& solution_, const rafko_mainframe::RafkoSettings& settings_)
  : settings(settings_)
  , execution_threads(settings.get_max_solve_threads())
  , net(rafko_net)
  , solution(solution_)
  , weights_in_partials(rafko_net.weight_table_size())
  , neurons_in_partials(solution.partial_solutions_size())
  {
  }

  /**
   * @brief      Copies the weights in the stored @RafkoNet reference into the provided solution.
   *             It supposes that the solution is one already built, and it is built from
   *             the same @RafkoNet referenced in the updater. Uses a different thread for every partial solution.
   */
  void update_solution_with_weights() const;

  /**
   * @brief      Copies the weights in the stored @RafkoNet reference into the provided solution.
   *             It supposes that the solution is one already built, and it is built from
   *             the same @RafkoNet referenced in the updater. Uses a different thread for every partial solution.
   *
   * @param[in]  weight_index   The index of the weight to take over fro the @RafkoNet
   */
  void update_solution_with_weight(std::uint32_t weight_index) const;

  /**
   * @brief      Provides a list of partials and weight indices inside them for every given network weight_index
   *             Each weight might be present inside multiple partials, but shall not be repeated multiple times
   *             inside each partial.
   *
   * @param[in]  network_weight_index   The Weight index inside the @RafkoNet
   *
   * @return     A vector of the following structure: {{partial index,weight_index},...,{...}}
   *             The elements of the vector are in ascending order by partial index.
   */
  PartialWeightPairs& get_relevant_partial_weight_indices_for(std::uint32_t network_weight_index) const;

  /**
   * @brief      Provides the partial index the given neuron_index belongs to
   *
   * @param[in]  neuron_index   The Neuron index inside the @RafkoNet
   *
   * @return     The index of the partial solution the Neuron belongs to
   */
  std::uint32_t get_relevant_partial_index_for(std::uint32_t neuron_index) const{
    return get_relevant_partial_index_for(neuron_index, solution, neurons_in_partials);
  }

  /**
   * @brief      Provides the partial index the given neuron_index belongs to
   *
   * @param[in]  neuron_index         The Neuron index inside the @RafkoNet
   * @param[in]  solution             The solution in which to search
   * @param      neurons_in_partials  The unordered map of the cache data
   *
   * @return     The index of the partial solution the Neuron belongs to
   */
  static std::uint32_t get_relevant_partial_index_for(
    std::uint32_t neuron_index, const rafko_net::Solution& solution,
    std::unordered_map<std::uint32_t, std::uint32_t>& neurons_in_partials
  );

  /**
   * @brief      Provides the first weight synapse index of the neuron inside the @PartialSolution
   *
   * @param[in]  neuron_index                       The Neuron index inside the @RafkoNet
   * @param[in]  partial                            The solution in which to search
   * @param      weight_synapse_starts_in_partial   The unordered map of the cache data
   *
   * @return     The index of the first weight synapse belonging to the given neuron
   */
  static std::uint32_t get_weight_synapse_start_index_in_partial(
    std::uint32_t neuron_index, const rafko_net::PartialSolution& partial,
    std::unordered_map<std::uint32_t, std::uint32_t>& weight_synapse_starts_in_partial
  );

#if(RAFKO_USES_OPENCL)
  /**
   * @brief      Provides the start index of the weight table for the given partial
   *             inside a weight common global weight table
   *
   * @param[in]  partial_index                The index of the Partial inside the solution
   * @param[in]  solution                     The solution in which to search
   * @param      weight_starts_in_partials    The unordered map of the cache data
   *
   * @return     The index of the first weight belonging to the partial solution inside the GPU device weight table
   */
  static std::uint32_t get_device_weight_table_start_for(
    std::uint32_t partial_index, const rafko_net::Solution& solution,
    std::unordered_map<std::uint32_t, std::uint32_t>& weight_starts_in_partials
  );
#endif/*(RAFKO_USES_OPENCL)*/

private:
  const rafko_mainframe::RafkoSettings& settings;
  rafko_utilities::ThreadGroup execution_threads;
  const rafko_net::RafkoNet& net;
  rafko_net::Solution& solution;

  mutable std::unordered_map<std::uint32_t,std::vector<std::pair<std::uint32_t,std::uint32_t>>> weights_in_partials; /* key: Weight index; {{partial_index, weight_index},...{..}} */
  mutable std::unordered_map<std::uint32_t, std::uint32_t> neurons_in_partials; /* key: Neuron index; value :Partial index */
  mutable std::mutex reference_mutex;

  /**
   * @brief      Copies the weights of a Neuron from the referenced @RafkoNet
   *             into the partial solution reference provided as an argument.
   *             The @PartialSolution must be built from the RafkoNet, as a pre-requisite.
   *
   * @param[in]  neuron_index                      The index of the Neuron inside the @SparsNet
   * @param      partial                           The partial solution to update
   * @param[in]  inner_neuron_weight_index_starts  The index in the weight table (of the @PartialSolution) where the inner neuron weights start
   */
  void copy_weights_of_neuron_to_partial_solution(
    std::uint32_t neuron_index, rafko_net::PartialSolution& partial, std::uint32_t inner_neuron_weight_index_starts
  ) const;
};

} /* namespace rafko_gym */

#endif /* RAFKO_WEIGHT_ADAPTER_H */

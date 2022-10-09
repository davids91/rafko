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

#ifndef FEATURE_GROUP_CACHE
#define FEATURE_GROUP_CACHE

#include "rafko_global.hpp"

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_net/services/synapse_iterator.hpp"

namespace rafko_net{

/**
 * @brief A container to keep track of different feature descriptors that need to be considered by the @NeuronRouter
 * to provide this information to whoever is using the collected subsets ( e.g.: @RafkoSolutionBuilder )
 */
class RAFKO_EXPORT FeatureGroupCache{
private:
  const std::uint32_t feature_group_index;
  const std::uint32_t num_of_neurons_needed;
  std::uint32_t num_of_neurons_solved = 0u;
public:
  const std::uint32_t checksum;

  FeatureGroupCache(const rafko_net::RafkoNet& network, std::uint32_t feature_group_index)
  : feature_group_index(feature_group_index)
  , num_of_neurons_needed(
    SynapseIterator<>(network.neuron_group_features(feature_group_index).relevant_neurons()).size()
  )
  , checksum(construct(network.neuron_group_features(feature_group_index)))
  {
  }

  void constexpr neuron_triggered(){
    ++num_of_neurons_solved;
  }

  bool constexpr solved() const{
     return (num_of_neurons_needed <= num_of_neurons_solved);
  }

  constexpr std::uint32_t get_index() const{
    return feature_group_index;
  }
private:
  /**
   * @brief      Calculate the checksum for the object and apply itself to every relevant neuron inside the cache
   *
   * @param[in]  host              A const reference of the feature descriptor this cache points to
   *
   * @return     The calculated checksum for the object
   */
  std::uint32_t construct(const FeatureGroup& host);
};

inline bool operator==(const FeatureGroupCache& a, const FeatureGroupCache& b){ return a.checksum == b.checksum; }

} /* namespace rafko_net */

#endif /* FEATURE_GROUP_CACHE */

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

#include "rafko_global.h"

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_net/services/synapse_iterator.h"

namespace rafko_net{

/**
 * @brief A container to keep track of different feature descriptors that need to be considered by the @NeuronRouter
 * to provide this information to whoever is using the collected subsets ( e.g.: @RafkoSolutionBuilder )
 */
class FeatureGroupCache{
private:
  const FeatureGroup& feature_in_network;
  const uint32 num_of_neurons_needed;
public:
  const uint32 checksum;

  FeatureGroupCache(const FeatureGroup& host)
  : feature_in_network(host)
  , num_of_neurons_needed(SynapseIterator<>(host.relevant_neurons()).size())
  , checksum(construct(host))
  { }

  void neuron_triggered(){ ++num_of_neurons_solved; }

  bool solved() const{ return (num_of_neurons_needed <= num_of_neurons_solved); }

  const FeatureGroup& get_host(){ return feature_in_network; }
private:
  uint32 num_of_neurons_solved = 0u;

  /**
   * @brief      Calculate the checksum for the object and apply itself to every relevant neuron inside the cache
   *
   * @param[in]  host              A const reference of the feature descriptor this cache points to
   *
   * @return     The calculated checksum for the object
   */
  uint32 construct(const FeatureGroup& host);
};

inline bool operator==(const FeatureGroupCache& a, const FeatureGroupCache& b){ return a.checksum == b.checksum; }

} /* namespace rafko_net */

#endif /* FEATURE_GROUP_CACHE */

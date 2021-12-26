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

#include "rafko_net/services/feature_group_cache.h"

namespace rafko_net {

uint32 FeatureGroupCache::construct(const FeatureGroup& host){
  uint32 calculated_checksum = 0u;
  uint32 fletchers_hash = 0; /* https://en.wikipedia.org/wiki/Fletcher%27s_checksum */
  SynapseIterator<> relevant_neurons = SynapseIterator<>(host.relevant_neurons());
  relevant_neurons.skim([this, &calculated_checksum, &fletchers_hash](IndexSynapseInterval interval){
    calculated_checksum |= interval.starts();
    calculated_checksum |= interval.interval_size();
    fletchers_hash |= calculated_checksum;
  });
  return( /* is this complexity too much? */
    ((calculated_checksum + 1) & 0x0000FFFFu)
    | ((fletchers_hash << 16) && 0xFFFF0000u)
  );
}

} /* namespace rafko_net */

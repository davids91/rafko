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

namespace rafko_gym{

rafko_net::RafkoNet RafkoNetworkDeltaChainLink::get_current_network(){
  if(network_built) return network;

  if(parent){
    network = parent->get_current_network();
  }else{
    network = original_network;
  }

  apply_to_network(data, network);
  return network;
}

void RafkoNetworkDeltaChainLink::apply_to_network(NetworkDeltaChainLinkData& delta, rafko_net::RafkoNet& network){
  if(0u == (delta.simple_changes_size() + delta.structural_changes_size())) return; /* No changes == nothing to do */
  std::uint32_t simple_changes_index = 0u;
  std::uint32_t structural_changes_index = 0u;
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
      ++simple_changes_index;
    }
  }

}


} /* namespace rafko_gym */

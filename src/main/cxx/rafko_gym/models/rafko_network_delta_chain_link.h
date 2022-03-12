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

#ifndef RAFKO_ENVIRONMENT_H
#define RAFKO_ENVIRONMENT_H

#include "rafko_global.h"

#include <memory>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/training.pb.h"

namespace RAFKO_FULL_EXPORT rafko_gym{

/**
 * @brief      A class representing one set of changes inside a network during training
 */
class RAFKO_FULL_EXPORT RafkoNetworkDeltaChainLink{
public:
  RafkoNetworkDeltaChainLink(
    const rafko_net::RafkoNet& original_network_, NetworkDeltaChainLinkData data_ = {},
    std::shared_ptr<RafkoNetworkDeltaChainLink> parent_ = std::shared_ptr<RafkoNetworkDeltaChainLink>()
  ):original_network(original_network_)
  , parent(parent_)
  , data(data_)
  { }

  const rafko_net::RafkoNet& get_original_network(){
    return original_network;
  }

  rafko_net::RafkoNet get_current_network();

  std::pair<std::unique_ptr<rafko_net::RafkoNet>,RafkoNetworkDeltaChainLink> create_new_chain(){
    std::unique_ptr<rafko_net::RafkoNet> current_network = std::make_unique<rafko_net::RafkoNet>(get_current_network());
    rafko_net::RafkoNet* current_network_ptr = current_network.get();
    return std::make_pair( std::move(current_network), RafkoNetworkDeltaChainLink(*current_network_ptr) );
  }

private:
  const rafko_net::RafkoNet& original_network;
  std::shared_ptr<RafkoNetworkDeltaChainLink> parent;
  NetworkDeltaChainLinkData data;
  rafko_net::RafkoNet network = rafko_net::RafkoNet();
  bool network_built = false;

  static void apply_to_network(NetworkDeltaChainLinkData& delta, rafko_net::RafkoNet& network);
};

} /* namespace rafko_gym */

#endif /* RAFKO_ENVIRONMENT_H */

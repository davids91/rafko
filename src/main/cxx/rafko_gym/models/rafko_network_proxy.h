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

#include "rafko_protocol/training.pb.h"
#include "rafko_gym/models/rafko_network_delta_chain_link.h"

namespace RAFKO_FULL_EXPORT rafko_gym{

/**
 * @brief      A class representing one view of a @RafkoNetworkDeltaChain
 */
class RAFKO_FULL_EXPORT RafkoNetworkProxy{
public:
  RafkoNetworkProxy(
    rafko_net::RafkoNet& original_network_,
    std::shared_pointer<RafkoNetworkDeltaChainLink> start_link = std::make_shared<RafkoNetworkDeltaChainLink>()
  );

  void stage(std::uint32_t weight_index, double weight_delta);
  void stage(std::vector<double>& weight_delta);
  void stage(RafkoNetworkDeltaChainLinkData&& weight_delta);
  std::shared_pointer<RafkoNetworkDeltaChainLink> commit();

  void revert();
  std::uint32_t get_version();
  rafko_net::RafkoNet& current_network();
  std::shared_pointer<RafkoNetworkDeltaChainLink> parent();

private:
  const rafko_net::RafkoNet& original_network;
  rafko_net::RafkoNet current_network = rafko_net::RafkoNet();
  std::shared_pointer<RafkoNetworkDeltaChainLink> current_link;
};

} /* namespace rafko_gym */

#endif /* RAFKO_ENVIRONMENT_H */

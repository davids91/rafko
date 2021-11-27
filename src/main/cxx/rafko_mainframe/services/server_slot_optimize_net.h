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

#ifndef SERVER_SLOT_OPTIMIZE_NET_H
#define SERVER_SLOT_OPTIMIZE_NET_H

#include "rafko_mainframe/services/server_slot.h"

namespace rafko_mainframe{

/**
 * @brief      This class describes a server slot which optimizes a Neural network
 *             for its stored datasets if the required input data and parameters are provided.
 */
class ServerSlotOptimizeNet : public ServerSlot{
public:
  void initialize(ServiceSlot&& service_slot_);
  void loop();
  void reset();
  void update_network(BuildNetworkRequest&& request);
  void update_network(RafkoNet&& net_);
  void accept_request(uint32 accept_request);
  SlotInfo get_info(uint32 request_bitstring);
  NeuralIOStream get_training_sample(uint32 sample_index, bool get_input, bool get_label) const;
  NeuralIOStream get_testing_sample(uint32 sample_index, bool get_input, bool get_label) const;
  ~ServerSlotOptimizeNet();
};

} /* namespace rafko_mainframe */

#endif /* SERVER_SLOT_OPTIMIZE_NET_H */

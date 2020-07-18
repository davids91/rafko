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

#ifndef SERVER_SLOT_APPROXIMIZE_NET_H
#define SERVER_SLOT_APPROXIMIZE_NET_H

#include "rafko_mainframe/services/server_slot.h"

namespace rafko_mainframe{


/**
 * @brief      This class describes a server slot which optimizes a Neural network
 *             for its stored datasets if the required input data and parameters are provided.
 */
class Server_slot_approximize_net : public Server_slot{
public:
  void initialize(Service_slot service_slot);
  void loop(void);
  void reset(void);
  void update_network(SparseNet net);
  void accept_request(Slot_request request);
  SparseNet get_network() const;
  Slot_response get_status() const;
  ~Server_slot_approximize_net();
};

} /* namespace rafko_mainframe */

#endif /* SERVER_SLOT_APPROXIMIZE_NET_H */
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

#ifndef SERVER_SLOT_FACTORY_H
#define SERVER_SLOT_FACTORY_H

#include <memory>

#include "rafko_protocol/deep_learning_service.pb.h"

#include "rafko_mainframe/services/server_slot.h"
#include "rafko_mainframe/services/server_slot_run_net.h"
#include "rafko_mainframe/services/server_slot_approximize_net.h"

namespace rafko_mainframe{

using std::unique_ptr;

/**
 * @brief      Front-end to create server slot objects.
 */
class ServerSlotFactory{
public:
  static unique_ptr<ServerSlot> build_server_slot(Slot_type slot_type){
    switch(slot_type){
      case serv_slot_to_run: return std::make_unique<ServerSlotRunNet>();
      case serv_slot_to_optimize: return std::make_unique<ServerSlotApproximizeNet>();
      default: throw std::runtime_error("Invalid or unsupported SLot type given to factory!");
    }
  }
};

} /* namespace rafko_mainframe */

#endif /* SERVER_SLOT_FACTORY_H */

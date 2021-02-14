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

#include "gen/deep_learning_service.pb.h"

#include "rafko_mainframe/services/server_slot.h"
#include "rafko_mainframe/services/server_slot_run_net.h"
#include "rafko_mainframe/services/server_slot_approximize_net.h"

namespace rafko_mainframe{

using std::unique_ptr;

/**
 * @brief      Front-end to create server slot objects.
 */
class Server_slot_factory{
public:
  static unique_ptr<Server_slot> build_server_slot(Slot_type slot_type){
    switch(slot_type){
      case SERV_SLOT_TO_RUN: return std::make_unique<Server_slot_run_net>();
      case SERV_SLOT_TO_APPROXIMIZE: return std::make_unique<Server_slot_approximize_net>();
      default: throw new std::runtime_error("Invalid or unsupported SLot type given to factory!");
    }
  }
};

} /* namespace rafko_mainframe */

#endif /* SERVER_SLOT_FACTORY_H */

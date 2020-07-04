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

#ifndef DEEP_LEARNING_SERVER_H
#define DEEP_LEARNING_SERVER_H

#include "sparse_net_global.h"
#include "gen/deep_learning_service.grpc.pb.h"

#include "models/server_slot.h"

namespace sparse_net_library{

/**
 * @brief      This class describes a server for deep learning related tasks. The supported operations are described in
 *             the @/proto/deep_learning_services.proto file. Functions defined in the service are thread-safe.
 */
class Deep_learning_server final : public Rafko_deep_learning::Service{
public:
  ~Deep_learning_server(void);
  ::grpc::Status add_slot(::grpc::ServerContext* context, const ::sparse_net_library::Service_slot* request, ::sparse_net_library::Slot_response* response);
  ::grpc::Status update_slot(::grpc::ServerContext* context, const ::sparse_net_library::Service_slot* request, ::sparse_net_library::Slot_response* response);
  ::grpc::Status request_action(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::sparse_net_library::Slot_response, ::sparse_net_library::Slot_request>* stream);
  ::grpc::Status get_network(::grpc::ServerContext* context, const ::sparse_net_library::Slot_request* request, ::sparse_net_library::SparseNet* response);

  /**
   * @brief      The main loop of the server to run to be able to provide the service
   */
  void loop(void);

private:
  vector<unique_ptr<Server_slot>> server_slots; /* points to different implementations of a @Server_slot */

};

} /* namespace sparse_net_library */

#endif /* DEEP_LEARNING_SERVER_H */

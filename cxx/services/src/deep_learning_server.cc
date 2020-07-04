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

#include "services/deep_learning_server.h"

#include "gen/common.pb.h"

namespace sparse_net_library{

void Deep_learning_server::loop(void){
  /* Go through the server slots */
  /* run each of them */
}

::grpc::Status Deep_learning_server::add_slot(::grpc::ServerContext* context, const ::sparse_net_library::Service_slot* request, ::sparse_net_library::Slot_response* response){
  return ::grpc::Status::OK;
}

::grpc::Status Deep_learning_server::update_slot(::grpc::ServerContext* context, const ::sparse_net_library::Service_slot* request, ::sparse_net_library::Slot_response* response){
  return ::grpc::Status::OK;
}

::grpc::Status Deep_learning_server::request_action(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::sparse_net_library::Slot_response, ::sparse_net_library::Slot_request>* stream){
  return ::grpc::Status::OK;
}

::grpc::Status Deep_learning_server::get_network(::grpc::ServerContext* context, const ::sparse_net_library::Slot_request* request, ::sparse_net_library::SparseNet* response){
  return ::grpc::Status::OK;
}


} /* namespace sparse_net_library */
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

#include "rafko_mainframe/services/deep_learning_server.h"

#include "gen/common.pb.h"

namespace rafko_mainframe{

void Deep_learning_server::loop(void){
  /* Go through the server slots */
  /* run each of them */
}

::grpc::Status Deep_learning_server::add_slot(::grpc::ServerContext* context, const ::rafko_mainframe::Service_slot* request, ::rafko_mainframe::Slot_response* response){
  return ::grpc::Status::OK;
}

::grpc::Status Deep_learning_server::update_slot(::grpc::ServerContext* context, const ::rafko_mainframe::Service_slot* request, ::rafko_mainframe::Slot_response* response){
  return ::grpc::Status::OK;
}

::grpc::Status Deep_learning_server::request_action(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::rafko_mainframe::Slot_response, ::rafko_mainframe::Slot_request>* stream){
  return ::grpc::Status::OK;
}

::grpc::Status Deep_learning_server::get_network(::grpc::ServerContext* context, const ::rafko_mainframe::Slot_request* request, ::sparse_net_library::SparseNet* response){
  std::cout  << "got it!" << std::endl;
  return ::grpc::Status::OK;
}

::grpc::Status Deep_learning_server::build_network(::grpc::ServerContext* context, const ::rafko_mainframe::Build_network_request* request, ::rafko_mainframe::Slot_response* response){
  return ::grpc::Status::OK;
}

::grpc::Status Deep_learning_server::build_one_neuron_network(::grpc::ServerContext* context, const ::rafko_mainframe::Build_network_request* request, ::rafko_mainframe::Slot_response* response){
  return ::grpc::Status::OK;
}


Deep_learning_server::~Deep_learning_server(void){
  server_slots.clear(); 
}


} /* namespace rafko_mainframe */
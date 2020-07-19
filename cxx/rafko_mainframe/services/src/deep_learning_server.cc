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
#include "rafko_mainframe/services/server_slot_factory.h"
#include "sparse_net_library/services/sparse_net_builder.h"

namespace rafko_mainframe{

using sparse_net_library::Sparse_net_builder;
using std::lock_guard;

void Deep_learning_server::loop(void){
  for(uint32 i = 0; i < server_slots.size(); ++i){
    if(is_server_slot_running[i]){
      server_slots[i]->loop();
    }
  }
}

::grpc::Status Deep_learning_server::add_slot(
  ::grpc::ServerContext* context, const ::rafko_mainframe::Service_slot* request,
  ::rafko_mainframe::Slot_response* response
){
  server_slots.push_back(std::move(Server_slot_factory::build_server_slot(request->type(),service_context)));
  server_slot_mutexs.push_back(std::make_unique<mutex>());
  is_server_slot_running.push_back(false);
  uint32 slot_id = find_id(request->slot_id());
  if((server_slots.size() > slot_id)&&(server_slots[slot_id])){
    lock_guard<mutex> my_lock(*server_slot_mutexs[slot_id]);
    server_slots[slot_id]->initialize(Service_slot(*request));
    response->CopyFrom(server_slots[slot_id]->get_status());
    return ::grpc::Status::OK;
  }else return ::grpc::Status::CANCELLED;
}

::grpc::Status Deep_learning_server::update_slot(
  ::grpc::ServerContext* context, const ::rafko_mainframe::Service_slot* request,
  ::rafko_mainframe::Slot_response* response
){
  uint32 slot_id = find_id(request->slot_id());
  if((server_slots.size() > slot_id)&&(server_slots[slot_id])){
    lock_guard<mutex> my_lock(*server_slot_mutexs[slot_id]);
    server_slots[slot_id]->initialize(Service_slot(*request));
    response->CopyFrom(server_slots[slot_id]->get_status());
    return ::grpc::Status::OK;
  }else return ::grpc::Status::CANCELLED;
}

::grpc::Status Deep_learning_server::build_network(
  ::grpc::ServerContext* context, const ::rafko_mainframe::Build_network_request* request,
  ::rafko_mainframe::Slot_response* response
){
  uint32 slot_id = find_id(request->target_slot_id());
  if((server_slots.size() > slot_id)&&(server_slots[slot_id])){
    lock_guard<mutex> my_lock(*server_slot_mutexs[slot_id]);
    if(0 < request->allowed_transfers_by_layer_size()){
      server_slots[slot_id]->update_network(SparseNet(
        *Sparse_net_builder().input_size(request->input_size())
        .expected_input_range(request->expected_input_range())
        .allowed_transfer_functions_by_layer({request->allowed_transfers_by_layer().begin(),request->allowed_transfers_by_layer().end()})
        .dense_layers({request->layer_sizes().begin(),request->layer_sizes().end()})
      ));
    }else{
      server_slots[slot_id]->update_network(SparseNet(
        *Sparse_net_builder().input_size(request->input_size())
        .expected_input_range(request->expected_input_range())
        .dense_layers({request->layer_sizes().begin(),request->layer_sizes().end()})
      ));
    }
    response->CopyFrom(server_slots[slot_id]->get_status());
    return ::grpc::Status::OK;
  }else return ::grpc::Status::CANCELLED;
}

::grpc::Status Deep_learning_server::request_action(
  ::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::rafko_mainframe::Slot_response,
  ::rafko_mainframe::Slot_request>* stream
){
  return ::grpc::Status::CANCELLED;
}

::grpc::Status Deep_learning_server::get_info(
  ::grpc::ServerContext* context, const ::rafko_mainframe::Slot_request* request,
  ::rafko_mainframe::Slot_info* response
){
  uint32 slot_id = find_id(request->target_slot_id());
  if((server_slots.size() > slot_id)&&(server_slots[slot_id])){
    lock_guard<mutex> my_lock(*server_slot_mutexs[slot_id]);
    response->CopyFrom(server_slots[slot_id]->get_info(*request));
    return ::grpc::Status::OK;
  }else return ::grpc::Status::CANCELLED;
  return ::grpc::Status::CANCELLED;
}


::grpc::Status Deep_learning_server::get_network(
  ::grpc::ServerContext* context, const ::rafko_mainframe::Slot_request* request,
  ::sparse_net_library::SparseNet* response
){
  uint32 slot_id = find_id(request->target_slot_id());
  if((server_slots.size() > slot_id)&&(server_slots[slot_id])){
    lock_guard<mutex> my_lock(*server_slot_mutexs[slot_id]);
    response->CopyFrom(server_slots[slot_id]->get_network());
    return ::grpc::Status::OK;
  }else return ::grpc::Status::CANCELLED;
}

uint32 Deep_learning_server::find_id(string id){
  uint32 index;
  for(index = 0; index < server_slots.size(); ++index)
    if((server_slots[index])&&(0 == id.compare(server_slots[index]->get_uuid())))
      break;
  return index;
}

} /* namespace rafko_mainframe */
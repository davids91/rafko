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

#include "rafko_mainframe/services/rafko_deep_learning_server.h"

#include <thread>

#include "rafko_protocol/common.pb.h"

#include "rafko_mainframe/services/server_slot_factory.h"

namespace rafko_mainframe{

using rafko_net::Transfer_functions;
using rafko_net::Transfer_functions_IsValid;
using std::lock_guard;
using std::thread;

void DeepLearningServer::loop(){
  lock_guard<mutex> my_lock(server_mutex);
  for(uint32 i = 0; i < server_slots.size(); ++i){
    if(0 < is_server_slot_running[i]){
      if(server_slot_mutexs[i]->try_lock()){ /* Able to lock the slot */
        thread loop_thread([&](const uint32 slot_index){ /* start a new thread for the loop operation */
          lock_guard<mutex> my_slot_lock_(*server_slot_mutexs[slot_index], std::adopt_lock);
          server_slots[slot_index]->loop();
          ++iteration[slot_index];
          lock_guard<mutex> my_cout_lock(server_mutex);
          std::cout << "\r";
          std::cout << "slot[" << slot_index << "]"
          << "; iteration:" << iteration[slot_index]
          << ": training error: "<<
            server_slots[slot_index]->get_info(slot_info_training_error).info_package(0)
          << "   " << std::endl;
          if(50000 < iteration[slot_index])is_server_slot_running[slot_index] = false;
        }, i);
        loop_thread.detach();
      } /* Unable to lock the slot, as it is busy, let's try again */
    }
  }
}

::grpc::Status DeepLearningServer::add_slot(
  ::grpc::ServerContext* context, const ::rafko_mainframe::ServiceSlot* request,
  ::rafko_mainframe::SlotResponse* response
){
  parameter_not_used(context);
  ::grpc::Status return_value = ::grpc::Status::CANCELLED;
  std::cout << " +++ add_slot +++ " << std::endl;
  try{
    lock_guard<mutex> my_lock(server_mutex);
    server_slot_mutexs.push_back(std::make_unique<mutex>());
    server_slots.push_back(unique_ptr<ServerSlot>(
      ServerSlotFactory::build_server_slot(request->type())
    ));
    is_server_slot_running.push_back(0);
    iteration.push_back(0);
    if(server_slots.back()){ /* If Item successfully added */
      lock_guard<mutex> my_lock(*server_slot_mutexs.back());
      server_slots.back()->initialize(ServiceSlot(*request));
      response->CopyFrom(server_slots.back()->get_status());
      return_value = ::grpc::Status::OK;
    }else return_value = ::grpc::Status::CANCELLED;
  }catch (const std::exception &exc){
    std::cout << "Exception while trying to add a new slot: " << exc.what() << std::endl;
    std::cout << exc.what() << std::endl;
    return_value = ::grpc::Status::CANCELLED;
  }
  std::cout << " --- add_slot --- " << std::endl;
  return return_value;
}

::grpc::Status DeepLearningServer::update_slot(
  ::grpc::ServerContext* context, const ::rafko_mainframe::ServiceSlot* request,
  ::rafko_mainframe::SlotResponse* response
){
  parameter_not_used(context);
  ::grpc::Status return_value = ::grpc::Status::CANCELLED;
  std::cout << " +++ update_slot +++ " << std::endl;
  try{
    const uint32 slot_index = find_id(request->slot_id());
    if((server_slots.size() > slot_index)&&(server_slots[slot_index])){
      lock_guard<mutex> my_lock(*server_slot_mutexs[slot_index]);
      std::cout << " on slot["<< slot_index <<"]" << std::endl;
      server_slots[slot_index]->initialize(ServiceSlot(*request));
      response->CopyFrom(server_slots[slot_index]->get_status());
      return_value = ::grpc::Status::OK;
    }else return_value = ::grpc::Status::CANCELLED;
  }catch (const std::exception &exc){
    std::cout << "Exception while trying to update slot: " << exc.what() << std::endl;
    std::cout << exc.what() << std::endl;
    return_value = ::grpc::Status::CANCELLED;
  }
  std::cout << " --- update_slot --- " << std::endl;
  return return_value;
}

::grpc::Status DeepLearningServer::ping(
  ::grpc::ServerContext* context, const ::rafko_mainframe::SlotRequest* request,
  ::rafko_mainframe::SlotResponse* response
){
  parameter_not_used(context);
  ::grpc::Status return_value = ::grpc::Status::CANCELLED;
  std::cout << " +++ ping +++ " << std::endl;
  try{
    const uint32 slot_index = find_id(request->target_slot_id());
    if((server_slots.size() > slot_index)&&(server_slots[slot_index])){
      lock_guard<mutex> my_lock(*server_slot_mutexs[slot_index]);
      std::cout << " on slot["<< slot_index <<"]" << std::endl;
      response->CopyFrom(server_slots[slot_index]->get_status());
    }
    return_value = ::grpc::Status::OK;
  }catch (const std::exception &exc){
    std::cout << "Exception while trying to ping: " << exc.what() << std::endl;
    std::cout << exc.what() << std::endl;
    return_value = ::grpc::Status::CANCELLED;
  }
  std::cout << " --- ping --- " << std::endl;
  return return_value;
}


::grpc::Status DeepLearningServer::build_network(
  ::grpc::ServerContext* context, const ::rafko_mainframe::BuildNetworkRequest* request,
  ::rafko_mainframe::SlotResponse* response
){
  parameter_not_used(context);
  ::grpc::Status return_value = ::grpc::Status::CANCELLED;
  std::cout << " +++ build_network +++ " << std::endl;
  try {
    const uint32 slot_index = find_id(request->target_slot_id());
    if((server_slots.size() > slot_index)&&(server_slots[slot_index])){
      std::cout << " on slot["<< slot_index <<"]" << std::endl;
      lock_guard<mutex> my_lock(*server_slot_mutexs[slot_index]);
      server_slots[slot_index]->update_network(BuildNetworkRequest(*request));
      response->CopyFrom(server_slots[slot_index]->get_status());
      return_value = ::grpc::Status::OK;
    }else return_value = ::grpc::Status::CANCELLED;
  }catch (const std::exception &exc){
    std::cout << std::endl << "Exception while trying to build a network: " << exc.what() << std::endl;
    std::cout << exc.what() << std::endl;
    return_value = ::grpc::Status::CANCELLED;
  }
  std::cout << " --- build_network --- " << std::endl;
  return return_value;
}

::grpc::Status DeepLearningServer::request_action(
  ::grpc::ServerContext* context,
  ::grpc::ServerReaderWriter< ::rafko_mainframe::SlotResponse,::rafko_mainframe::SlotRequest>* stream
){
  parameter_not_used(context);
  ::grpc::Status return_value = ::grpc::Status::CANCELLED;
  std::cout << " +++ request_action +++ " << std::endl;
  try{
    SlotRequest current_request;
    while(stream->Read(&current_request)){
      const uint32 slot_index = find_id(current_request.target_slot_id());
      if((server_slots.size() > slot_index)&&(server_slots[slot_index])){
        uint32 request_bitstring = current_request.request_bitstring();
        uint32 request_index = current_request.request_index();
        lock_guard<mutex> my_lock(*server_slot_mutexs[slot_index]);
        std::cout << " on slot["<< slot_index <<"]" << std::endl;
        if(0 < (request_bitstring & serv_slot_to_start)){
          is_server_slot_running[slot_index] = 1;
        }
        if(0 < (request_bitstring & serv_slot_to_stop)){
          is_server_slot_running[slot_index] = 0;
          iteration[slot_index] = 0;
        }
        if(0 < (request_bitstring & serv_slot_to_reset)){
          server_slots[slot_index]->reset();
        }
        if(0 < (request_bitstring & serv_slot_to_takeover_net)){
          std::cout << " --- request_action --- " << std::endl;
          return ::grpc::Status::CANCELLED; /* Not implemented yet */
        }
        if(0 < (request_bitstring & serv_slot_to_takeover_training_set)){
          std::cout << " --- request_action --- " << std::endl;
          return ::grpc::Status::CANCELLED; /* Not implemented yet */
        }
        if(0 < (request_bitstring & serv_slot_to_append_test_set)){
          std::cout << " --- request_action --- " << std::endl;
          return ::grpc::Status::CANCELLED; /* Not implemented yet */
        }
        if(0 < (request_bitstring & serv_slot_to_distill_network)){
          std::cout << " --- request_action --- " << std::endl;
          return ::grpc::Status::CANCELLED; /* Not implemented yet */
        }
        if(0 < (request_bitstring & serv_slot_to_amplify_network)){
          std::cout << " --- request_action --- " << std::endl;
          return ::grpc::Status::CANCELLED; /* Not implemented yet */
        }
        if(0 < (request_bitstring & serv_slot_to_get_training_sample)){
          SlotResponse response(server_slots[slot_index]->get_status());
          response.mutable_data_stream()->CopyFrom(
            server_slots[slot_index]->get_training_sample(request_index, true, true)
          );
          stream->Write(response);
        }
        if(0 < (request_bitstring & serv_slot_to_get_test_sample)){
          SlotResponse response(server_slots[slot_index]->get_status());
          response.mutable_data_stream()->CopyFrom(
            server_slots[slot_index]->get_testing_sample(request_index, true, true)
          );
          stream->Write(response);
        }
        if(0 < (request_bitstring & serv_slot_to_refresh_solution)){
          server_slots[slot_index]->accept_request(request_bitstring & serv_slot_to_refresh_solution);
        }
        if(0 < (request_bitstring & serv_slot_to_run_once)){
          SlotResponse response(server_slots[slot_index]->get_status());
          if(0 == current_request.data_stream().input_size()){ /* No input data specified */
            response.mutable_data_stream()->CopyFrom(server_slots[slot_index]->run_net_once(
              server_slots[slot_index]->get_training_sample(request_index, true, false)
            ));
          }else{ /* Some input data is specified, run the network based on the data provided from the request */
            response.mutable_data_stream()->CopyFrom(
              server_slots[slot_index]->run_net_once(current_request.data_stream())
            );
          }
          stream->Write(response);
        }
        if(0 < (request_bitstring & serv_slot_to_die)){
          lock_guard<mutex> my_lock(server_mutex);
          std::cout << " --- request_action --- " << std::endl;
          return ::grpc::Status::CANCELLED; /* Not implemented yet */
        }
      }else{
        std::cout << " --- request_action --- " << std::endl;
        return ::grpc::Status::CANCELLED; /* Server slot not found */
      }
    }/* Until all the requests are processed */
    return_value = ::grpc::Status::OK;
  }catch (const std::exception &exc){
    std::cout << "Exception while trying to request an action: " << exc.what() << std::endl;
    std::cout << exc.what() << std::endl;
    return_value = ::grpc::Status::CANCELLED;
  }
  std::cout << " --- request_action --- " << std::endl;
  return return_value;
}

::grpc::Status DeepLearningServer::get_info(
  ::grpc::ServerContext* context, const ::rafko_mainframe::SlotRequest* request,
  ::rafko_mainframe::SlotInfo* response
){
  parameter_not_used(context);
  ::grpc::Status return_value = ::grpc::Status::CANCELLED;
  std::cout << " +++ get_info +++ " << std::endl;
  try{
    const uint32 slot_index = find_id(request->target_slot_id());
    if((server_slots.size() > slot_index)&&(server_slots[slot_index])){
      lock_guard<mutex> my_lock(*server_slot_mutexs[slot_index]);
      std::cout << " on slot["<< slot_index <<"]" << std::endl;
      response->CopyFrom(server_slots[slot_index]->get_info(request->request_bitstring()));
      return_value = ::grpc::Status::OK;
    }else return_value = ::grpc::Status::CANCELLED;
  }catch (const std::exception &exc){
    std::cout << "Exception while trying to get info: " << exc.what() << std::endl;
    std::cout << exc.what() << std::endl;
    return_value = ::grpc::Status::CANCELLED;
  }
  std::cout << " --- get_info --- " << std::endl;
  return return_value;
}


::grpc::Status DeepLearningServer::get_network(
  ::grpc::ServerContext* context, const ::rafko_mainframe::SlotRequest* request,
  ::rafko_net::RafkoNet* response
){
  parameter_not_used(context);
  ::grpc::Status return_value = ::grpc::Status::CANCELLED;
  std::cout << " +++ get_network +++ " << std::endl;
  try{
    const uint32 slot_index = find_id(request->target_slot_id());
    if((server_slots.size() > slot_index)&&(server_slots[slot_index])){
      lock_guard<mutex> my_lock(*server_slot_mutexs[slot_index]);
      std::cout << " on slot["<< slot_index <<"]" << std::endl;
      response->CopyFrom(server_slots[slot_index]->get_network());
      return_value = ::grpc::Status::OK;
    }else return_value = ::grpc::Status::CANCELLED;
  }catch (const std::exception &exc){
    std::cout << "Exception while trying to provide network: " << exc.what() << std::endl;
    std::cout << exc.what() << std::endl;
    return_value = ::grpc::Status::CANCELLED;
  }
  std::cout << " --- get_network --- " << std::endl;
  return return_value;
}

uint32 DeepLearningServer::find_id(string id){
  uint32 index;
  for(index = 0; index < server_slots.size(); ++index)
    if((server_slots[index])&&(0 == id.compare(server_slots[index]->get_uuid())))
      break;
  return index;
}

} /* namespace rafko_mainframe */

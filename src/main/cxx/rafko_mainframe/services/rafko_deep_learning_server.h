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

#include "rafko_global.h"

#include <string>
#include <mutex>

#include "rafko_protocol/deep_learning_service.grpc.pb.h"

#include "rafko_mainframe/services/server_slot.h"

namespace rafko_mainframe{

using std::vector;
using std::unique_ptr;
using std::shared_ptr;
using std::string;
using std::mutex;

/**
 * @brief      This class describes a server for deep learning related tasks. The supported operations are described in
 *             the @/proto/deep_learning_services.proto file. Functions defined in the service are thread-safe.
 */
class DeepLearningServer final : public RafkoDeepLearning::Service{
public:
  DeepLearningServer()
  :  server_slots()
  ,  server_slot_mutexs()
  ,  is_server_slot_running()
  ,  iteration()
  ,  server_mutex()
  { }

  DeepLearningServer(const DeepLearningServer& other) = delete;/* Copy constructor */
  DeepLearningServer(DeepLearningServer&& other) = delete; /* Move constructor */
  DeepLearningServer& operator=(const DeepLearningServer& other) = delete; /* Copy assignment */
  DeepLearningServer& operator=(DeepLearningServer&& other) = delete; /* Move assignment */

  ::grpc::Status add_slot(::grpc::ServerContext* context, const ::rafko_mainframe::ServiceSlot* request, ::rafko_mainframe::SlotResponse* response);
  ::grpc::Status update_slot(::grpc::ServerContext* context, const ::rafko_mainframe::ServiceSlot* request, ::rafko_mainframe::SlotResponse* response);

  ::grpc::Status ping(::grpc::ServerContext* context, const ::rafko_mainframe::SlotRequest* request, ::rafko_mainframe::SlotResponse* response);
  ::grpc::Status build_network(::grpc::ServerContext* context, const ::rafko_mainframe::BuildNetworkRequest* request, ::rafko_mainframe::SlotResponse* response);
  ::grpc::Status request_action(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::rafko_mainframe::SlotResponse, ::rafko_mainframe::SlotRequest>* stream);
  ::grpc::Status get_info(::grpc::ServerContext* context, const ::rafko_mainframe::SlotRequest* request, ::rafko_mainframe::SlotInfo* response);
  ::grpc::Status get_network(::grpc::ServerContext* context, const ::rafko_mainframe::SlotRequest* request, ::rafko_net::RafkoNet* response);

  /**
   * @brief      The main loop of the server to run to be able to provide the service
   */
  void loop();
  ~DeepLearningServer(){ server_slots.clear(); }

private:
  vector<unique_ptr<ServerSlot>> server_slots; /* points to different implementations of a @ServerSlot */
  vector<unique_ptr<mutex>> server_slot_mutexs;
  vector<uint8> is_server_slot_running;
  vector<uint32> iteration;
  mutex server_mutex; /* Aims to protect modification of the state of the server ( mainly ServerSlots ) */

  /**
   * @brief      Tries to find the index of the server slot with the given identifier
   *
   * @param[in]  id    The identifier
   *
   * @return     Index of the slot in @server_slots
   */
  uint32 find_id(string id);
};

} /* namespace rafko_mainframe */

#endif /* DEEP_LEARNING_SERVER_H */

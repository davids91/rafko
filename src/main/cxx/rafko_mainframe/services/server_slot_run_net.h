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

#ifndef SERVER_SLOT_RUN_NET_H
#define SERVER_SLOT_RUN_NET_H

#include <memory>
#include <string>

#include "sparse_net_library/services/solution_solver.h"

#include "rafko_mainframe/services/server_slot.h"

namespace rafko_mainframe{

using std::vector;
using sparse_net_library::SparseNet;
using sparse_net_library::Solution;
using sparse_net_library::Solution_solver;

using std::string;
using std::unique_ptr;

/**
 * @brief      This class describes a server slot which runs a Neural network
 *             if the required input data and parameters are provided
 */
class Server_slot_run_net : public Server_slot{
public:
  Server_slot_run_net()
  :  Server_slot()
  ,  network()
  ,  network_solution()
  ,  network_solver()
  { 
    service_slot->set_type(SERV_SLOT_TO_RUN);
    network = google::protobuf::Arena::CreateMessage<SparseNet>(context.get_arena_ptr());
  }

  void initialize(Service_slot&& service_slot_);
  void refresh_solution(void);
  Neural_io_stream run_net_once(const Neural_io_stream& data_stream);
  ~Server_slot_run_net(void){ }

  /* Inlinable interfaces */
  void update_network(Build_network_request&& request){
    expose_state();
    service_slot->set_state(service_slot->state() | SERV_SLOT_MISSING_NET);

    if((0 == request.layer_sizes_size())||(request.layer_sizes_size() != request.allowed_transfers_by_layer_size()))
      throw new std::runtime_error("Invalid network build request!");

    if(get_uuid() == request.target_slot_id()){
      *network = *build_network_from_request(std::move(request));
      if(nullptr != network) service_slot->set_state(service_slot->state() & ~SERV_SLOT_MISSING_NET);
      refresh_solution();
    }
    finalize_state();
  }

  void update_network(SparseNet&& net_){
    expose_state();
    if(0 < net_.neuron_array_size()){
      *network = std::move(net_);
      service_slot->set_state(service_slot->state() & ~SERV_SLOT_MISSING_NET);
      refresh_solution();
    }
    finalize_state();
  }

  void reset(void){
    update_network(SparseNet());
  }

  Slot_info get_info(uint32 request_bitstring){
    return Slot_info(); /* No info to be provided */
  }
  
  SparseNet get_network(void) const{
    return *network;
  }

  Slot_response get_status(void) const{
    Slot_response ret;
    ret.set_slot_id(service_slot->slot_id());
    ret.set_slot_state(service_slot->state());
    return ret;
  }

  /* Not supported interfaces */
  void loop(void){
    throw new std::runtime_error("Loop operation not supported in a network runner slot!");
  }

  void accept_request(uint32 request_bitstring){
    throw new std::runtime_error("Direct Requests not supported in a network runner slot!");
  }

  Neural_io_stream get_training_sample(uint32 sample_index, bool get_input, bool get_label) const{
    throw new std::runtime_error("Data sets not supported in a network runner slot!");
  }

  Neural_io_stream get_testing_sample(uint32 sample_index, bool get_input, bool get_label) const{
    throw new std::runtime_error("Data sets not supported in a network runner slot!");
  }

protected:
  SparseNet* network;
  Solution* network_solution;
  unique_ptr<Solution_solver> network_solver;
};

} /* namespace rafko_mainframe */

#endif /* SERVER_SLOT_RUN_NET_H */

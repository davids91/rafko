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

#include "rafko_net/services/solution_solver.h"

#include "rafko_mainframe/services/server_slot.h"

namespace rafko_mainframe{

using std::vector;
using rafko_net::RafkoNet;
using rafko_net::Solution;
using rafko_net::SolutionSolver;

using std::string;
using std::unique_ptr;

/**
 * @brief      This class describes a server slot which runs a Neural network
 *             if the required input data and parameters are provided
 */
class ServerSlotRunNet : public ServerSlot{
public:
  ServerSlotRunNet()
  :  ServerSlot()
  ,  network()
  ,  network_solution()
  ,  network_solver()
  {
    service_slot->set_type(serv_slot_to_run);
    network = google::protobuf::Arena::CreateMessage<RafkoNet>(context.get_arena_ptr());
  }

  void initialize(ServiceSlot&& service_slot_);
  void refresh_solution(void);
  NeuralIOStream run_net_once(const NeuralIOStream& data_stream);
  ~ServerSlotRunNet(void){ }

  /* Inlinable interfaces */
  void update_network(BuildNetworkRequest&& request){
    expose_state();
    service_slot->set_state(service_slot->state() | serv_slot_missing_net);

    if((0 == request.layer_sizes_size())||(request.layer_sizes_size() != request.allowed_transfers_by_layer_size()))
      throw std::runtime_error("Invalid network build request!");

    if(get_uuid() == request.target_slot_id()){
      *network = *build_network_from_request(std::move(request));
      if(nullptr != network) service_slot->set_state(service_slot->state() & ~serv_slot_missing_net);
      refresh_solution();
    }
    finalize_state();
  }

  void update_network(RafkoNet&& net_){
    expose_state();
    if(0 < net_.neuron_array_size()){
      *network = std::move(net_);
      service_slot->set_state(service_slot->state() & ~serv_slot_missing_net);
      refresh_solution();
    }
    finalize_state();
  }

  void reset(void){
    update_network(RafkoNet());
  }

  SlotInfo get_info(uint32 request_bitstring){
    parameter_not_used(request_bitstring);
    return SlotInfo(); /* No info to be provided */
  }

  RafkoNet get_network(void) const{
    return *network;
  }

  SlotResponse get_status(void) const{
    SlotResponse ret;
    ret.set_slot_id(service_slot->slot_id());
    ret.set_slot_state(service_slot->state());
    return ret;
  }

  /* Not supported interfaces */
  void loop(void){
    throw std::runtime_error("Loop operation not supported in a network runner slot!");
  }

  void accept_request(uint32 request_bitstring){
    parameter_not_used(request_bitstring);
    throw std::runtime_error("Direct Requests not supported in a network runner slot!");
  }

  NeuralIOStream get_training_sample(uint32 sample_index, bool get_input, bool get_label) const{
    parameter_not_used(sample_index);
    parameter_not_used(get_input);
    parameter_not_used(get_label);
    throw std::runtime_error("Data sets not supported in a network runner slot!");
  }

  NeuralIOStream get_testing_sample(uint32 sample_index, bool get_input, bool get_label) const{
    parameter_not_used(sample_index);
    parameter_not_used(get_input);
    parameter_not_used(get_label);
    throw std::runtime_error("Data sets not supported in a network runner slot!");
  }

protected:
  RafkoNet* network;
  Solution* network_solution;
  unique_ptr<SolutionSolver> network_solver;
};

} /* namespace rafko_mainframe */

#endif /* SERVER_SLOT_RUN_NET_H */

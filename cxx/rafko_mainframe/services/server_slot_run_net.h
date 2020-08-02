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

#include "gen/solution.pb.h"
#include "sparse_net_library/services/solution_builder.h"
#include "sparse_net_library/services/solution_solver.h"
#include "rafko_mainframe/models/service_context.h"

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
  Server_slot_run_net(Service_context& context_)
  :  context(context_)
  ,  network_input()
  ,  network()
  ,  network_solution()
  ,  network_solver()
  { service_slot.set_type(SERV_SLOT_TO_RUN); }

  void initialize(Service_slot&& service_slot_);
  void update_network(SparseNet&& net_);
  void refresh_solution(void);

  void reset(void){
    update_network(SparseNet());
  }

  Neural_io_stream run_net_once(const Neural_io_stream& data_stream);

  Slot_info get_info(uint32 request_bitstring){
    return Slot_info(); /* No info to be provided */
  }
  
  SparseNet get_network(void) const{
    return network;
  }

  Slot_response get_status(void) const{
    Slot_response ret;
    ret.set_slot_id(service_slot.slot_id());
    ret.set_slot_state(service_slot.state());
    return ret;
  }


  ~Server_slot_run_net(void){ }

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
  Service_context& context;
  vector<sdouble32> network_input;
  SparseNet network;
  Solution network_solution;
  unique_ptr<Solution_solver> network_solver;
};

} /* namespace rafko_mainframe */

#endif /* SERVER_SLOT_RUN_NET_H */

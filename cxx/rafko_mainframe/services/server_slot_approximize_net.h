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

#ifndef SERVER_SLOT_APPROXIMIZE_NET_H
#define SERVER_SLOT_APPROXIMIZE_NET_H

#include "rafko_mainframe/services/server_slot_run_net.h"
#include "sparse_net_library/models/cost_function.h"
#include "sparse_net_library/models/data_aggregate.h"
#include "sparse_net_library/services/sparse_net_approximizer.h"

namespace rafko_mainframe{

using sparse_net_library::Cost_function;
using sparse_net_library::Data_aggregate;
using sparse_net_library::Sparse_net_approximizer;

using std::unique_ptr;
using std::shared_ptr;

/**
 * @brief      This class describes a server slot which optimizes a Neural network
 *             for its stored datasets if the required input data and parameters are provided.
 */
class Server_slot_approximize_net : public Server_slot_run_net{
public:
  Server_slot_approximize_net(Service_context context_)
  :  Server_slot_run_net(context_)
  ,  cost_function()
  ,  training_set()
  ,  test_set()
  ,  network_approximizer()
  ,  iteration(0)
  { }

  void initialize(Service_slot&& service_slot_);
  void loop(void);
  void reset(void);
  void update_network(SparseNet&& net_);
  void accept_request(Slot_request&& request_);
  Slot_info get_info(Slot_request request);
  ~Server_slot_approximize_net(void){
    network_approximizer.reset();
    test_set.reset();
    training_set.reset();
    cost_function.reset();
  }

private:
  shared_ptr<Cost_function> cost_function;
  shared_ptr<Data_aggregate> training_set;
  shared_ptr<Data_aggregate> test_set;
  unique_ptr<Sparse_net_approximizer> network_approximizer;

  uint32 iteration;
};

} /* namespace rafko_mainframe */

#endif /* SERVER_SLOT_APPROXIMIZE_NET_H */

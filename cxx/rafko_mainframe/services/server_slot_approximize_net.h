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

#include "sparse_net_library/models/cost_function.h"
#include "sparse_net_library/models/data_aggregate.h"
#include "sparse_net_library/services/sparse_net_approximizer.h"

#include "rafko_mainframe/services/server_slot_run_net.h"

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
  Server_slot_approximize_net()
  :  Server_slot_run_net()
  ,  cost_function()
  ,  training_set()
  ,  test_set()
  ,  network_approximizer()
  ,  iteration(0)
  { service_slot->set_type(SERV_SLOT_TO_APPROXIMIZE); }

  void initialize(Service_slot&& service_slot_);
  void accept_request(uint32 request_bitstring);
  Slot_info get_info(uint32 request_bitstring);

  void update_network(Build_network_request&& request){
    Server_slot_run_net::update_network(std::move(request));
    update_cost_function();
    update_trainer();
  }

  void update_network(SparseNet&& net_){
    Server_slot_run_net::update_network(std::move(net_));
    update_cost_function();
    update_trainer();
  }

  void loop(void){
    if(SERV_SLOT_OK == service_slot->state()){
      network_approximizer->collect_fragment();
      network_approximizer->apply_fragment();
      ++iteration;
    }else throw new std::runtime_error("Loop called of an invalid server slot!");
  }

  void reset(void){
    if(0 < service_slot->state()){
      training_set->reset_errors();
      test_set->reset_errors();
      network_approximizer->discard_fragment();
    }else throw new std::runtime_error("Reset called of an invalid server slot!");
  }

  Neural_io_stream get_training_sample(uint32 sample_index, bool get_input, bool get_label) const{
    if((training_set)&&(0 < service_slot->state())){
      Neural_io_stream result;
      result.set_sequence_size(training_set->get_sequence_size());
      if(get_input)result.set_input_size(training_set->get_input_sample(0).size());
      if(get_label)result.set_label_size(training_set->get_label_sample(0).size());
      /*!Note: since all inputs should have the same size, 0 is okay here */
      if(get_input || get_label)get_data_sample(training_set, sample_index, result);
      return result;
    }else throw std::runtime_error("Invalid training set queried for sample!");
  }

  Neural_io_stream get_test_sample(uint32 sample_index, bool get_input, bool get_label) const{
    if((training_set)&&(0 < service_slot->state())){
      Neural_io_stream result;
      result.set_sequence_size(test_set->get_sequence_size());
      if(get_input)result.set_input_size(test_set->get_input_sample(0).size());
      if(get_label)result.set_label_size(test_set->get_label_sample(0).size());
      /*!Note: since all inputs should have the same size, 0 is okay here */
      if(get_input || get_label)get_data_sample(test_set, sample_index, result);
      return result;
    }else throw std::runtime_error("Invalid training set queried for sample!");
  }

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

  void update_cost_function(void);
  void update_trainer(void);
};

} /* namespace rafko_mainframe */

#endif /* SERVER_SLOT_APPROXIMIZE_NET_H */

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

#include "rafko_net/models/cost_function.h"
#include "rafko_gym/models/data_aggregate.h"
#include "rafko_gym/services/rafko_environment_data_set.h"
#include "rafko_gym/services/rafko_net_approximizer.h"

#include "rafko_mainframe/services/server_slot_run_net.h"

namespace rafko_mainframe{

/**
 * @brief      This class describes a server slot which optimizes a Neural network
 *             for its stored datasets if the required input data and parameters are provided.
 */
class ServerSlotApproximizeNet : public ServerSlotRunNet{
public:
  ServerSlotApproximizeNet()
  :  ServerSlotRunNet()
  { service_slot->set_type(serv_slot_to_optimize); }

  void initialize(ServiceSlot&& service_slot_);
  void accept_request(uint32 request_bitstring);
  SlotInfo get_info(uint32 request_bitstring);

  void update_network(BuildNetworkRequest&& request){
    ServerSlotRunNet::update_network(std::move(request));
    update_cost_function();
    update_trainer();
  }

  void update_network(rafko_net::RafkoNet&& net_){
    ServerSlotRunNet::update_network(std::move(net_));
    update_cost_function();
    update_trainer();
  }

  void loop(){
    if(serv_slot_ok == service_slot->state()){
      network_approximizer->collect_approximates_from_weight_gradients();
      network_approximizer->apply_fragment();
      ++iteration;
    }else throw std::runtime_error("Loop called of an invalid server slot!");
  }

  void reset(){
    if(0 < service_slot->state()){
      training_set->reset_errors();
      test_set->reset_errors();
      network_approximizer->discard_fragment();
    }else throw std::runtime_error("Reset called of an invalid server slot!");
  }

  NeuralIOStream get_training_sample(uint32 sample_index, bool get_input, bool get_label) const{
    if((training_set)&&(0 < service_slot->state())){
      NeuralIOStream result;
      result.set_sequence_size(training_set->get_sequence_size());
      if(get_input)result.set_input_size(training_set->get_input_sample(0).size());
      if(get_label)result.set_label_size(training_set->get_label_sample(0).size());
      /*!Note: since all inputs should have the same size, 0 is okay here */
      if(get_input || get_label)get_data_sample(training_set, sample_index, result);
      return result;
    }else throw std::runtime_error("Invalid training set queried for sample!");
  }

  NeuralIOStream get_test_sample(uint32 sample_index, bool get_input, bool get_label) const{
    if((training_set)&&(0 < service_slot->state())){
      NeuralIOStream result;
      result.set_sequence_size(test_set->get_sequence_size());
      if(get_input)result.set_input_size(test_set->get_input_sample(0).size());
      if(get_label)result.set_label_size(test_set->get_label_sample(0).size());
      /*!Note: since all inputs should have the same size, 0 is okay here */
      if(get_input || get_label)get_data_sample(test_set, sample_index, result);
      return result;
    }else throw std::runtime_error("Invalid training set queried for sample!");
  }

  ~ServerSlotApproximizeNet(){
    network_approximizer.reset();
    test_set.reset();
    training_set.reset();
    cost_function.reset();
  }

private:
  std::shared_ptr<rafko_net::CostFunction> cost_function;
  std::shared_ptr<rafko_gym::DataAggregate> training_set;
  std::shared_ptr<rafko_gym::DataAggregate> test_set;
  std::shared_ptr<rafko_gym::RafkoEnvironmentDataSet> environment_data_set;
  std::unique_ptr<rafko_gym::RafkoNetApproximizer> network_approximizer;

  uint32 iteration = 0;

  void update_cost_function();
  void update_trainer();
};

} /* namespace rafko_mainframe */

#endif /* SERVER_SLOT_APPROXIMIZE_NET_H */

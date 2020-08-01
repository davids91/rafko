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

#include "rafko_mainframe/services/server_slot_approximize_net.h"

#include "sparse_net_library/services/sparse_net_builder.h"
#include "sparse_net_library/services/function_factory.h"

namespace rafko_mainframe{

using sparse_net_library::Solution_builder;
using sparse_net_library::Function_factory;
using sparse_net_library::COST_FUNCTION_UNKNOWN;
using sparse_net_library::WEIGHT_UPDATER_UNKNOWN;

using std::make_unique;

void Server_slot_approximize_net::initialize(Service_slot&& service_slot_){
  service_slot.set_type(service_slot_.type());
  if(SERV_SLOT_TO_APPROXIMIZE != service_slot_.type())
    throw new std::runtime_error("Incorrect Server slot initialization!");
  else{
    service_slot.set_state(0u); /* Reset state */
    /* ####################################################################
     * NEURAL NETWORK
     * #################################################################### */
    if(0 < network.neuron_array_size())
      update_network(std::move(*service_slot_.mutable_network()));
    /* ####################################################################
     * COST FUNCTION
     * #################################################################### */
    if(COST_FUNCTION_UNKNOWN != service_slot_.cost_function()){
      service_slot.set_state(service_slot.state() | SERV_SLOT_MISSING_COST_FUNCTION);
      if(cost_function) cost_function.reset();
      cost_function = std::move(Function_factory::build_cost_function(
        network, service_slot_.cost_function(), context
      ));
      if(cost_function){
        service_slot.set_cost_function(service_slot_.cost_function());
        service_slot.set_state(service_slot.state() & ~SERV_SLOT_MISSING_COST_FUNCTION);
      }
    }/* if(COST_FUNCTION_UNKNOWN != service_slot_.cost_function()) */
    /* ####################################################################
     * DATA SETS
     * #################################################################### */
    service_slot.set_state(service_slot.state() | SERV_SLOT_MISSING_DATA_SET);
    if(
      (cost_function) /* Data aggreaget requires a cost function */
      &&(0 < service_slot_.training_set().inputs_size())
    ){
      if(training_set)training_set.reset();
      training_set = std::make_shared<Data_aggregate>(*service_slot_.mutable_training_set(), cost_function);
      if(training_set)
        service_slot.set_state(service_slot.state() & ~SERV_SLOT_MISSING_DATA_SET);
    }
    if(
      (cost_function) /* Data aggreaget requires a cost function */
      &&(0 < service_slot_.test_set().inputs_size())
    ){
      if(test_set)test_set.reset();
      if(
        (service_slot_.training_set().inputs_size() == service_slot_.test_set().inputs_size())
        &&(service_slot_.training_set().labels_size() == service_slot_.test_set().labels_size())
      ){
        training_set = std::make_shared<Data_aggregate>(*service_slot_.mutable_test_set(), cost_function);
      }else test_set = training_set;
    }
    /* ####################################################################
     * TRAINER
     * #################################################################### */
    if(service_slot_.has_hypers())
      (void)context.set_hypers(Service_hyperparameters());
    service_slot.set_state(service_slot.state() | SERV_SLOT_MISSING_TRAINER);
    if(network_approximizer)
      network_approximizer.reset();
    if(WEIGHT_UPDATER_UNKNOWN != service_slot.weight_updater()){
      network_approximizer = std::make_unique<Sparse_net_approximizer>(
        network, *training_set, *test_set, service_slot_.weight_updater(), context
      );
      if(network_approximizer){
        service_slot.set_state(service_slot.state() & ~SERV_SLOT_MISSING_TRAINER);
        service_slot.set_weight_updater(service_slot_.weight_updater());
      }
    }
    finalize_state();
  }
}

void Server_slot_approximize_net::update_network(SparseNet&& net_){
  Server_slot_run_net::update_network(std::move(net_));
  expose_state();
  if(network_approximizer)
    network_approximizer.reset();
  service_slot.set_state(service_slot.state() | SERV_SLOT_MISSING_TRAINER);
  if(WEIGHT_UPDATER_UNKNOWN != service_slot.weight_updater()){
    network_approximizer = std::make_unique<Sparse_net_approximizer>(
      network, *training_set, *test_set, service_slot.weight_updater(), context
    );
    if(network_approximizer)
      service_slot.set_state(service_slot.state() & ~SERV_SLOT_MISSING_TRAINER);
  }
  finalize_state();
}

void Server_slot_approximize_net::accept_request(Slot_request&& request_){
  throw new std::runtime_error("Unimplemented operation!");
}

Slot_info Server_slot_approximize_net::get_info(Slot_request request){
  Slot_info response;
  if(training_set){
    if(0 < (request.request_bitstring() & SLOT_INFO_TRAINING_ERROR)){
      response.add_info_field(SLOT_INFO_TRAINING_ERROR);
      response.add_info_package(training_set->get_error());
    }
    if(0 < (request.request_bitstring() & SLOT_INFO_TRAINING_SET_SEQUENCE_COUNT)){
      response.add_info_field(SLOT_INFO_TRAINING_SET_SEQUENCE_COUNT);
      response.add_info_package(training_set->get_number_of_sequences());
    }
  }
  if(test_set){
    if(0 < (request.request_bitstring() & SLOT_INFO_TEST_ERROR)){
      response.add_info_field(SLOT_INFO_TEST_ERROR);
      response.add_info_package(test_set->get_error());
    }
    if(0 < (request.request_bitstring() & SLOT_INFO_TEST_SET_SEQUENCE_COUNT)){
      response.add_info_field(SLOT_INFO_TEST_SET_SEQUENCE_COUNT);
      response.add_info_package(test_set->get_number_of_sequences());
    }
  }
  return response;
}

} /* rafko_mainframe */
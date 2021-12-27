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

#include "rafko_net/services/rafko_net_builder.h"
#include "rafko_net/services/solution_builder.h"
#include "rafko_net/services/function_factory.h"

namespace rafko_mainframe{

void ServerSlotApproximizeNet::initialize(ServiceSlot&& service_slot_){
  if(serv_slot_to_optimize != service_slot_.type()) throw std::runtime_error("Incorrect Server slot initialization!");
  else{
    /* ####################################################################
     * Initial state checks
     * #################################################################### */
    expose_state();
    if( /* In case any set pointers are invalid, or points to empty sets */
      (!training_set)||(0 == training_set->get_number_of_sequences())
      ||(!test_set)||(0 == test_set->get_number_of_sequences()) /* invalidate state */
    )service_slot->set_state(service_slot->state() | serv_slot_missing_data_set);
    if( /* In case there is an available training set to be copied from the request */
      (service_slot_.has_training_set())
      &&(0 < service_slot_.training_set().inputs_size())
    )*service_slot->mutable_training_set() = service_slot_.training_set();
    if( /* In case there is an available test set to be copied from the request */
      (service_slot_.has_test_set())
      &&(0 < service_slot_.test_set().inputs_size())
    )*service_slot->mutable_test_set() = service_slot_.test_set();
    if(
      (rafko_gym::Weight_updaters_IsValid(service_slot_.weight_updater()))
      &&(rafko_gym::weight_updater_unknown != service_slot_.weight_updater())
    )service_slot->set_weight_updater(service_slot_.weight_updater());
    if(service_slot_.has_hypers())(void)settings.set_hypers(service_slot_.hypers());

    /* ####################################################################
     * NEURAL NETWORK
     * #################################################################### */
    if(0 < service_slot_.network().neuron_array_size()){ /* If the given network is not empty */
      update_network(std::move(*service_slot_.mutable_network()));
    }else if(0 == service_slot->network().neuron_array_size()){
      service_slot->set_state(service_slot->state() | serv_slot_missing_net);
    }

    /* ####################################################################
     * COST FUNCTION
     * #################################################################### */
    if(
      rafko_net::Cost_functions_IsValid(service_slot_.cost_function())
      &&(rafko_net::cost_function_unknown != service_slot_.cost_function())
    )service_slot->set_cost_function(service_slot_.cost_function());
    update_cost_function();
    expose_state();

    /* ####################################################################
     * DATA SETS
     * #################################################################### */
    if(
      (cost_function) /* Data aggregate requires a cost function */
      &&(0 < service_slot->training_set().inputs_size())
    ){
      service_slot->set_state(service_slot->state() | serv_slot_missing_data_set);
      if(training_set)training_set.reset();
      training_set = std::make_shared<rafko_gym::DataAggregate>(settings, *service_slot->mutable_training_set(), cost_function);
      if(training_set)
        service_slot->set_state(service_slot->state() & ~serv_slot_missing_data_set);
      service_slot->set_state(service_slot->state() | serv_slot_missing_trainer); /* data set have changed, trainer needs to be re-initialized */
    }
    if(
      (cost_function) /* Data aggregate requires a cost function */
      &&(0 < service_slot_.test_set().inputs_size())
    ){
      if(test_set)test_set.reset();
      if(
        (service_slot_.training_set().inputs_size() == service_slot_.test_set().inputs_size())
        &&(service_slot_.training_set().labels_size() == service_slot_.test_set().labels_size())
      ){
        training_set = std::make_shared<rafko_gym::DataAggregate>(settings, *service_slot_.mutable_test_set(), cost_function);
      }else test_set = training_set;
      service_slot->set_state(service_slot->state() | serv_slot_missing_trainer); /* data set have changed, trainer needs to be re-initialized */
    }else test_set = training_set;
    /* ####################################################################
     * TRAINER
     * #################################################################### */
    update_trainer();
    finalize_state();
  }
}

void ServerSlotApproximizeNet::accept_request(uint32 request_bitstring){
  if(0 < (request_bitstring & serv_slot_to_refresh_solution)){
    refresh_solution();
  }
}

SlotInfo ServerSlotApproximizeNet::get_info(uint32 request_bitstring){
  SlotInfo response;
  if(0 < (request_bitstring & slot_info_iteration)){
    response.add_info_field(slot_info_iteration);
    response.add_info_package(iteration);
  }
  if(training_set){
    if(0 < (request_bitstring & slot_info_training_error)){
      response.add_info_field(slot_info_training_error);
      response.add_info_package(training_set->get_error_avg());
    }
    if(0 < (request_bitstring & slot_info_training_set_sequence_count)){
      response.add_info_field(slot_info_training_set_sequence_count);
      response.add_info_package(training_set->get_number_of_sequences());
    }
  }
  if(test_set){
    if(0 < (request_bitstring & slot_info_test_error)){
      response.add_info_field(slot_info_test_error);
      response.add_info_package(test_set->get_error_avg());
    }
    if(0 < (request_bitstring & slot_info_test_set_sequence_count)){
      response.add_info_field(slot_info_test_set_sequence_count);
      response.add_info_package(test_set->get_number_of_sequences());
    }
  }
  return response;
}

void ServerSlotApproximizeNet::update_cost_function(){
  expose_state();
  if(
    (cost_function)
    &&( /* in case there's a cost function object, but an invalid cost function type stored */
      (!rafko_net::Cost_functions_IsValid(service_slot->cost_function()))
      ||(rafko_net::cost_function_unknown == service_slot->cost_function())
      ||(service_slot->cost_function() != cost_function->get_type())
    ) /* or the stored cost function type doesn't match the objects */
  )cost_function.reset();
  if(!cost_function)service_slot->set_state(service_slot->state() | serv_slot_missing_cost_function);
  if(
    (rafko_net::cost_function_unknown != service_slot->cost_function())
    &&(
      (0 == (service_slot->state() & serv_slot_missing_net))
      ||(0 == (service_slot->state() & serv_slot_missing_data_set))
      ||(0 < service_slot->training_set().feature_size())
    )
  ){
    if(cost_function) cost_function.reset();
    service_slot->set_state(service_slot->state() | serv_slot_missing_cost_function);
    cost_function = std::move(rafko_net::FunctionFactory::build_cost_function( service_slot->cost_function(), settings ));
    if(cost_function){
      service_slot->set_state(service_slot->state() & ~serv_slot_missing_cost_function);
    }
  }
  finalize_state();
}

void ServerSlotApproximizeNet::update_trainer(){

  expose_state();
  if(
    (rafko_gym::Weight_updaters_IsValid(service_slot->weight_updater()))
    &&(rafko_gym::weight_updater_unknown != service_slot->weight_updater())
    &&(0 == (service_slot->state() & serv_slot_missing_data_set))
    &&(0 == (service_slot->state() & serv_slot_missing_net))
  ){
    if(network_approximizer)network_approximizer.reset();
    service_slot->set_state(service_slot->state() | serv_slot_missing_trainer);
    service_slot->set_weight_updater(service_slot->weight_updater());
    if(environment_data_set){
      environment_data_set.reset();
    }
    environment_data_set = std::make_shared<rafko_gym::RafkoEnvironmentDataSet>(
      settings, *training_set, *test_set
    );
    network_approximizer = std::make_unique<rafko_gym::RafkoNetApproximizer>(
      settings, *network, *environment_data_set, service_slot->weight_updater()
    );
    if(network_approximizer){
      service_slot->set_state(service_slot->state() & ~serv_slot_missing_trainer);
      service_slot->set_weight_updater(service_slot->weight_updater());
    }
  }else{
    if(network_approximizer)network_approximizer.reset();
    service_slot->set_state(service_slot->state() | serv_slot_missing_trainer);
  }
  finalize_state();
}

} /* rafko_mainframe */

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

#include "rafko_mainframe/services/server_slot_run_net.h"

#include "sparse_net_library/services/sparse_net_builder.h"

namespace rafko_mainframe{

using sparse_net_library::Solution_builder;

void Server_slot_run_net::initialize(Service_slot&& service_slot_){
  service_slot.set_type(service_slot_.type());
  if(SERV_SLOT_TO_RUN != service_slot.type())
    throw new std::runtime_error("Incorrect Server slot initialization!");
  else{
    service_slot.set_slot_id(generate_uuid());
    service_slot.set_state(0u); /* Reset state and update accordingly */
    if(0 < network.neuron_array_size()){
      update_network(std::move(*service_slot.mutable_network()));
    }
  }
}

void Server_slot_run_net::refresh_solution(void){
  expose_state();
  service_slot.set_state(service_slot.state() | SERV_SLOT_MISSING_SOLUTION);
  if(0 < network.neuron_array_size()){
    network_solution = *Solution_builder(context).build(network);
    network_solver = std::make_unique<Solution_solver>(network_solution, context);
    service_slot.set_state(service_slot.state() & ~SERV_SLOT_MISSING_SOLUTION);
  }else service_slot.set_state(service_slot.state() | SERV_SLOT_MISSING_NET);
  finalize_state();
}

void Server_slot_run_net::update_network(SparseNet&& net_){
  expose_state();
  std::cout << "{run1:"<< service_slot.state() <<"}";
  if(0 < net_.neuron_array_size()){
    network = std::move(net_);
    service_slot.set_state(service_slot.state() & ~SERV_SLOT_MISSING_NET);
    std::cout << "{run2:"<< service_slot.state() <<"}";
    refresh_solution();
  }
  std::cout << "{run3:"<< service_slot.state() <<"}";     
  finalize_state();
}

Neural_io_stream Server_slot_run_net::run_net_once(const Neural_io_stream& data_stream){
  if(
    (SERV_SLOT_OK == service_slot.state())
    ||(0 == (SERV_SLOT_MISSING_SOLUTION & service_slot.state()))
  ){
    Neural_io_stream result;
    uint32 sequence_start_index = 0;
    vector<sdouble32> result_package = vector<sdouble32>( /* reserve enough space for the result.. */
      network.output_neuron_number() * data_stream.sequence_size()
    ); /* ..which is the output of the network multiplied by the sequence size */
    for(uint32 sequence = 0; sequence < data_stream.sequence_size(); ++sequence){
      network_solver->solve({ /* Solve based on input */
        data_stream.package().begin() + sequence_start_index,
        data_stream.package().begin() + sequence_start_index + data_stream.feature_size(),
      });
      std::copy(
        network_solver->get_neuron_data(0).end() - network_solver->get_output_size(),
        network_solver->get_neuron_data(0).end(),
        result_package.begin() + sequence_start_index
      );
      sequence_start_index += data_stream.feature_size();
    } /* for every sequence */

    /* Compiling the result into the argument */
    result.set_feature_size(network.output_neuron_number());
    result.set_sequence_size(data_stream.sequence_size());
    *result.mutable_package() = {result_package.begin(), result_package.end()};
    return result;
  }throw new std::runtime_error("Invalid attached network run attempt!");
}

} /* rafko_mainframe */
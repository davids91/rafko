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

#include "rafko_utilities/models/const_vector_subrange.h"
#include "rafko_net/services/rafko_net_builder.h"
#include "rafko_net/services/solution_builder.h"

namespace rafko_mainframe{

void ServerSlotRunNet::initialize(ServiceSlot&& service_slot_){
  service_slot->set_type(service_slot_.type());
  if(serv_slot_to_run != service_slot->type())
    throw std::runtime_error("Incorrect Server slot initialization!");
  else{
    service_slot->set_slot_id(generate_uuid());
    service_slot->set_state(0u); /* Reset state and update accordingly */
    if(0 < network->neuron_array_size()){
      update_network(std::move(*service_slot->mutable_network()));
    }
  }
}

void ServerSlotRunNet::refresh_solution(){
  expose_state();
  service_slot->set_state(service_slot->state() | serv_slot_missing_solution);
  if(0 < network->neuron_array_size()){
    network_solution = rafko_net::SolutionBuilder(settings).build(*network);
    network_solver = std::unique_ptr<rafko_net::SolutionSolver>(rafko_net::SolutionSolver::Builder(*network_solution, settings).build());
    service_slot->set_state(service_slot->state() & ~serv_slot_missing_solution);
  }else service_slot->set_state(service_slot->state() | serv_slot_missing_net);
  finalize_state();
}

NeuralIOStream ServerSlotRunNet::run_net_once(const NeuralIOStream& data_stream){
  if(
    (serv_slot_ok == service_slot->state())
    ||(0 == (serv_slot_missing_solution & service_slot->state()))
  ){
    NeuralIOStream result;
    uint32 sequence_start_index = 0;
    std::vector<sdouble32> result_package = std::vector<sdouble32>( /* reserve enough space for the result.. */
      network->output_neuron_number() * data_stream.sequence_size()
    ); /* ..which is the output of the network multiplied by the sequence size */
    std::vector<sdouble32> input;
    for(uint32 sequence = 0; sequence < data_stream.sequence_size(); ++sequence){
      input = { /* Copy the input data to a temporary */
        data_stream.package().begin() + sequence_start_index,
        data_stream.package().begin() + sequence_start_index + data_stream.input_size(),
      };
      rafko_utilities::ConstVectorSubrange<> neuron_data = network_solver->solve(input,true,0);
      std::copy(
        neuron_data.end() - network_solution->output_neuron_number(),
        neuron_data.end(),
        result_package.begin() + sequence_start_index
      );
      sequence_start_index += data_stream.feature_size();
    } /* for every sequence */
    /* Compiling the result into the argument */
    result.set_feature_size(network->output_neuron_number());
    result.set_sequence_size(data_stream.sequence_size());
    *result.mutable_package() = {result_package.begin(), result_package.end()};
    return result;
  }throw std::runtime_error("Invalid attached network run attempt!");
}

} /* rafko_mainframe */

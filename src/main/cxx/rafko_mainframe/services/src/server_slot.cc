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

#include "rafko_mainframe/services/server_slot.h"

#include <random>
#include <memory>
#include <vector>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_net/services/rafko_net_builder.h"

namespace rafko_mainframe{

std::string ServerSlot::generate_uuid(){
  static std::random_device dev;
  static std::mt19937 rng(dev());

  std::uniform_int_distribution<int> dist(0, 15);

  const char *v = "0123456789abcdef";
  const bool dash[] = { 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0 };

  std::string res;
  for (int i = 0; i < 16; i++) {
      if (dash[i]) res += "-";
      res += v[dist(rng)];
      res += v[dist(rng)];
  }
  return res;
}

SlotResponse ServerSlot::get_status() const{
  SlotResponse response;
  response.set_slot_id(get_uuid());
  response.set_slot_state(service_slot->state());
  return response;
}


std::string ServerSlot::get_uuid() const{
  if(0 != service_slot->slot_id().compare("")) return service_slot->slot_id();
    else throw std::runtime_error("Empty UUID is queried!");
}

void ServerSlot::get_data_sample(std::shared_ptr<rafko_gym::DataAggregate> data_set, uint32 sample_index, NeuralIOStream& target) const{
  if( /* In case the attached training set is valid */
    (data_set) /* Avoid nullpointers */
    &&(sample_index < data_set->get_number_of_label_samples()) /* Avoid out of bounds */
    &&(data_set->get_sequence_size() == target.sequence_size()) /* Sequence sizes should match as a safeguard, or should it?  */
    &&(0 == target.package_size()) /* Only copy data into empty packages */
  ){ /* And the index is not out of bounds */
    const uint32 number_of_input_arrays = data_set->get_sequence_size() + data_set->get_prefill_inputs_number();
    target.mutable_package()->Reserve( /* Reserve the needed space for the element */
      (target.input_size() * number_of_input_arrays)
      + (target.label_size() * target.sequence_size())
    );

    uint32 inputs_index = (sample_index * number_of_input_arrays); /* Add the input into the field */
    if(data_set->get_input_sample(inputs_index).size() == target.input_size()){ /* If the sizes match for the input field */
      for(uint32 sequence_iterator = 0; sequence_iterator < number_of_input_arrays; ++sequence_iterator){
        std::copy(
          data_set->get_input_sample(inputs_index).begin(),
          data_set->get_input_sample(inputs_index).end(),
          RepeatedFieldBackInserter(target.mutable_package())
        );
        ++inputs_index;
      }
    }

    uint32 labels_index = (sample_index * data_set->get_sequence_size()); /* Add the label into the field */
    if(data_set->get_label_sample(labels_index).size() == target.label_size()){ /* If the sizes match for the input field */
      for(uint32 sequence_iterator = 0; sequence_iterator < target.sequence_size(); ++sequence_iterator){
        std::copy(
          data_set->get_label_sample(labels_index).begin(),
          data_set->get_label_sample(labels_index).end(),
          RepeatedFieldBackInserter(target.mutable_package())
        );
        ++labels_index;
      }
    }
    target.set_feature_size(0); /* Not filling that up this time.. */
  }
}

rafko_net::RafkoNet* ServerSlot::build_network_from_request(BuildNetworkRequest&& request){
  if(0 < request.allowed_transfers_by_layer_size()){
    uint32 layer_index = 0;
    std::vector<std::vector<rafko_net::Transfer_functions>> allowed_transfers(request.allowed_transfers_by_layer_size());
    for(const sint32& allowed : request.allowed_transfers_by_layer()){
      if(rafko_net::Transfer_functions_IsValid(static_cast<rafko_net::Transfer_functions>(allowed)))
        allowed_transfers[layer_index++] = std::vector<rafko_net::Transfer_functions>(1, static_cast<rafko_net::Transfer_functions>(allowed));
      else throw std::runtime_error("Unknown transfer function detected!");
    }
    return rafko_net::RafkoNetBuilder(settings).input_size(request.input_size())
      .expected_input_range(request.expected_input_range())
      .allowed_transfer_functions_by_layer(allowed_transfers)
      .dense_layers({request.layer_sizes().begin(),request.layer_sizes().end()});
  }else{
    return rafko_net::RafkoNetBuilder(settings).input_size(request.input_size())
      .expected_input_range(request.expected_input_range())
      .dense_layers({request.layer_sizes().begin(),request.layer_sizes().end()});
  }
}

} /* rafko_mainframe */

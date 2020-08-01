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

namespace rafko_mainframe{

using std::string;
using std::random_device;
using std::mt19937;
using std::uniform_int_distribution;

string Server_slot::generate_uuid(void){
  static random_device dev;
  static mt19937 rng(dev());

  uniform_int_distribution<int> dist(0, 15);

  const char *v = "0123456789abcdef";
  const bool dash[] = { 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0 };

  string res;
  for (int i = 0; i < 16; i++) {
      if (dash[i]) res += "-";
      res += v[dist(rng)];
      res += v[dist(rng)];
  }
  return res;
}

Slot_response Server_slot::get_status(void) const{
  Slot_response response;
  response.set_slot_id(get_uuid());
  response.set_slot_state(service_slot.state());
  return response;
}


string Server_slot::get_uuid(void) const{
  if(0 != service_slot.slot_id().compare("")) return service_slot.slot_id();
    else throw new std::runtime_error("Empty UUID is queried!");
}

Neural_io_stream Server_slot::get_data_sample(shared_ptr<Data_aggregate> data_set, uint32 sample_index) const{
  Neural_io_stream result; /* Create the resulting message and set header data for it */
  if( /* In case the attached training set is valid */
    (data_set)
    &&(sample_index < data_set->get_number_of_samples())
  ){ /* And the index is not out of bounds */
    uint32 filled_numbers = 0;
    result.set_feature_size(0);
    result.set_input_size(data_set->get_input_sample(sample_index).size());
    result.set_label_size(data_set->get_label_sample(sample_index).size());
    result.set_sequence_size(data_set->get_sequence_size());
    result.mutable_package()->Resize( /* Reserve the needed space for the element */
      (
        (result.input_size() * result.sequence_size()) 
        + (result.label_size() * result.sequence_size())
      ), double_literal(0)
    );

    /* Add the input into the field */
    for(uint32 sequence_iterator = 0; sequence_iterator < result.sequence_size(); ++sequence_iterator){
      result.mutable_package()[filled_numbers + (result.input_size() * result.sequence_size())] = {
        data_set->get_input_sample(sample_index + sequence_iterator).begin(),
        data_set->get_input_sample(sample_index + sequence_iterator).end()
      };
      filled_numbers += result.input_size();
    }

    /* Add the label into the field */
    for(uint32 sequence_iterator = 0; sequence_iterator < result.sequence_size(); ++sequence_iterator){
      result.mutable_package()[filled_numbers + (result.label_size() * result.sequence_size())] = {
        data_set->get_label_sample(sample_index + sequence_iterator).begin(),
        data_set->get_label_sample(sample_index + sequence_iterator).end()
      };
      filled_numbers += result.input_size();
    }
  }
  return result;
}

} /* rafko_mainframe */
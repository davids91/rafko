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

#include "test/catch.hpp"

#include "sparse_net_global.h"
#include "sparse_net_library/models/data_ringbuffer.h"

#include <vector>

namespace sparse_net_library_test {

using std::vector;
using std::copy;

using sparse_net_library::Data_ringbuffer;
using sparse_net_library::Input_synapse_interval;

/*###############################################################################################
 * Testing Ringbuffer implementation by creating a ringbuffer object and adding new entries in
 *  multiple times, and checking the validity of the data.
 * */
void check_data_match(vector<sdouble32>& sample_data, vector<sdouble32>& ringbuffer_data){
  REQUIRE(sample_data.size() == ringbuffer_data.size());
  for(uint32 i = 0; i < sample_data.size(); ++i){
    CHECK(sample_data[i] == ringbuffer_data[i]);
  }
}

TEST_CASE("Testing Data Ringbuffer implementation", "[data-handling]"){
  uint32 buffer_number = 5;
  uint32 buffer_size = 30;
  vector<sdouble32> data_sample(buffer_size, double_literal(0.0));
  vector<sdouble32> previous_data_sample(buffer_size, double_literal(0.0));
  Data_ringbuffer buffer(buffer_number, buffer_size);

  REQUIRE( buffer.buffer_size() == buffer_size );
  REQUIRE( buffer.get_sequence_size() == buffer_number );

  /* By default every data should be 0 */
  for(uint32 i = 0; i<buffer_number; ++i)
    check_data_match(data_sample, buffer.get_element(i));

  /* Adding numbers */
  for(uint32 variant = 0; variant < (buffer_number*2); ++variant){
    check_data_match(data_sample, buffer.get_element(0));
    check_data_match(previous_data_sample, buffer.get_element(1));
    copy(data_sample.begin(),data_sample.end(), previous_data_sample.begin());
    buffer.step();
    for(uint32 b = 0; b < buffer_size; ++b){
      data_sample[b] += b;
      buffer.get_element(0)[b] = data_sample[b];
    }
  }
}

/*###############################################################################################
 * Testing a sequence of runs to be stored in the ringbuffer, and seeing if the indexing is as expected
 *  by querying sequence indices and comparing to past reaches
 *  Used interfaces: 
 * - get_sequence_size
 * - get_const_element
 * - get_sequence_index
 * */
TEST_CASE("Testing if ringbuffer past indexing logic is as expected", "[data-handling]"){
  uint32 sequence_number = 5;
  uint32 buffer_size = 30;
  Data_ringbuffer buffer(sequence_number, buffer_size);
  Input_synapse_interval input_synapse;
  vector<sdouble32> data_sample(buffer_size);

  /* Simulate some runs: each element in the buffer shall have the value of it's past value */
  for(sint32 i = sequence_number-1; i >= 0; --i){
    buffer.step();
    for(sdouble32& sample_element : data_sample)
      sample_element = i;
    copy(data_sample.begin(),data_sample.end(), buffer.get_element(0).begin());
  }

  /*!Note: To understand Sequential indexes in the Data ringbuffer, this might help: 
  for(sint32 i = sequence_number-1; i >= 0; --i)
    std::cout << "[" << i << "]-";
  std::cout << "past index (buffer conents also in this example)" << std::endl;
  for(uint32 i = 0; i < sequence_number; i++)
    std::cout << "[" << i << "]-";
  std::cout << "sequence index" << std::endl; */

  /* See if the first sequence reach back only to that index */
  input_synapse.set_reach_past_loops(0);
  CHECK( static_cast<sint32>(buffer.get_sequence_size()) > buffer.get_sequence_index(0, input_synapse) );
  CHECK( static_cast<sint32>(sequence_number-1) == buffer.get_sequence_index(0, input_synapse) );

  for(uint32 i = 1; i < sequence_number; ++i){
    input_synapse.set_reach_past_loops(i);
    CHECK( static_cast<sint32>(buffer.get_sequence_size()) <= buffer.get_sequence_index(0, input_synapse) );
  }

  /* See if later sequences reach back to the relevant index */
  for(uint32 sequence_iterator = 1; sequence_iterator < sequence_number; ++sequence_iterator){
    for(uint32 reach_back_count = 0; reach_back_count <= sequence_iterator; ++reach_back_count){
      input_synapse.set_reach_past_loops(reach_back_count);
      CHECK( static_cast<sint32>(buffer.get_sequence_size()) > buffer.get_sequence_index(sequence_iterator, input_synapse) );
      CHECK( static_cast<sint32>((sequence_number - sequence_iterator - 1) + reach_back_count) == buffer.get_sequence_index(sequence_iterator, input_synapse) );
      CHECK( buffer.get_const_element(buffer.get_sequence_index(sequence_iterator, input_synapse))[0] == buffer.get_sequence_index(sequence_iterator, input_synapse) );
      CHECK( buffer.get_const_element(sequence_iterator,input_synapse,0) == buffer.get_sequence_index(sequence_iterator, input_synapse) );
      CHECK( buffer.get_const_element(sequence_iterator,input_synapse)[0] == buffer.get_sequence_index(sequence_iterator, input_synapse) );
    }
  }
}

} /* namespace sparse_net_library_test */

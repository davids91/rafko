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

#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_utilities/models/data_ringbuffer.h"

#include "test/test_utility.h"

namespace rafko_utilities_test {

/*###############################################################################################
 * Testing Ringbuffer implementation by creating a ringbuffer object and adding new entries in
 *  multiple times, and checking the validity of the data.
 * */
void check_data_match(std::vector<sdouble32>& sample_data, std::vector<sdouble32>& ringbuffer_data){
  REQUIRE(sample_data.size() == ringbuffer_data.size());
  for(uint32 i = 0; i < sample_data.size(); ++i){
    CHECK(sample_data[i] == ringbuffer_data[i]);
  }
}

TEST_CASE("Testing Data Ringbuffer implementation", "[data-handling]"){
  rafko_mainframe::RafkoSettings settings;
  uint32 buffer_number = 5;
  uint32 buffer_size = 30;
  std::vector<sdouble32> data_sample(buffer_size, double_literal(0.0));
  std::vector<sdouble32> previous_data_sample(buffer_size, double_literal(0.0));
  rafko_utilities::DataRingbuffer buffer(buffer_number, buffer_size);

  REQUIRE( buffer.buffer_size() == buffer_size );
  REQUIRE( buffer.get_sequence_size() == buffer_number );

  /* By default every data should be 0 */
  for(uint32 i = 0; i<buffer_number; ++i)
    check_data_match(data_sample, buffer.get_element(i));

  /* Adding numbers */
  for(uint32 variant = 0; variant < (buffer_number*2); ++variant){
    check_data_match(data_sample, buffer.get_element(0));
    check_data_match(previous_data_sample, buffer.get_element(1));
    std::copy(data_sample.begin(),data_sample.end(), previous_data_sample.begin());
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
  rafko_mainframe::RafkoSettings settings;
  uint32 sequence_number = 5;
  uint32 buffer_size = 30;
  rafko_utilities::DataRingbuffer buffer(sequence_number, buffer_size);
  std::vector<sdouble32> data_sample(buffer_size);

  /* Simulate some runs: each element in the buffer shall have the value of it's past value */
  for(sint32 i = sequence_number-1; i >= 0; --i){
    buffer.step();
    for(sdouble32& sample_element : data_sample)
      sample_element = i;
    std::copy(data_sample.begin(),data_sample.end(), buffer.get_element(0).begin());
  }

  /*!Note: To understand Sequential indexes in the Data ringbuffer, this might help:
  for(sint32 i = sequence_number-1; i >= 0; --i)
    std::cout << "[" << i << "]-";
  std::cout << "past index (buffer conents also in this example)" << std::endl;
  for(uint32 i = 0; i < sequence_number; i++)
    std::cout << "[" << i << "]-";
  std::cout << "sequence index" << std::endl; */
}

} /* namespace rafko_utilities_test */

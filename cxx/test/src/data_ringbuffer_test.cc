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
#include "models/data_ringbuffer.h"

#include <vector>

namespace sparse_net_library_test {

using std::vector;
using sparse_net_library::uint32;
using sparse_net_library::sdouble32;
using sparse_net_library::Data_ringbuffer;

/*###############################################################################################
 * Testing Ringbuffer implementation by creating a ringbuffer object and adding new entries in
 * multiple times, and checking the validity of the data.
 * */
void check_data_match(vector<sdouble32>& sample_data, vector<sdouble32>& ringbuffer_data){
  REQUIRE(sample_data.size() == ringbuffer_data.size());
  for(uint32 i = 0; i < sample_data.size(); ++i){
    CHECK(sample_data[i] == ringbuffer_data[i]);
  }
}

TEST_CASE( "Testing Data Ringbuffer implementation", "[data-handling]" ) {
  uint32 buffer_number = 5;
  uint32 buffer_size = 30;
  vector<sdouble32> data_sample(buffer_size, double_literal(0.0));
  vector<sdouble32> previous_data_sample(buffer_size, double_literal(0.0));
  Data_ringbuffer buffer(buffer_number, buffer_size);

  /* By default every data should be 0 */
  for(uint32 i = 0; i<buffer_number; ++i)
    check_data_match(data_sample, buffer.get_element(i));

  /* Adding numbers */
  for(uint32 variant = 0; variant < (buffer_number*2); ++variant){
    check_data_match(data_sample, buffer.get_element(0));
    check_data_match(previous_data_sample, buffer.get_element(1));
    std::copy(
      data_sample.begin(),data_sample.end(), previous_data_sample.begin()
    );
    buffer.step();
    for(uint32 b = 0; b < buffer_size; ++b){
      data_sample[b] += b;
      buffer.get_element(0)[b] = data_sample[b];
    }
  }

}

} /* namespace sparse_net_library_test */

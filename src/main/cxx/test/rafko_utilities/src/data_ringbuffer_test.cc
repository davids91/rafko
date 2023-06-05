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

#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "rafko_mainframe/models/rafko_settings.hpp"
#include "rafko_utilities/models/data_ringbuffer.hpp"

#include "test/test_utility.hpp"

namespace rafko_utilities_test {

/*###############################################################################################
 * Testing Ringbuffer implementation by creating a ringbuffer object and adding
 * new entries in multiple times, and checking the validity of the data.
 * */
void check_data_match(std::vector<double> &sample_data,
                      std::vector<double> &ringbuffer_data) {
  REQUIRE(sample_data.size() == ringbuffer_data.size());
  for (std::uint32_t i = 0; i < sample_data.size(); ++i) {
    CHECK(sample_data[i] == ringbuffer_data[i]);
  }
}

TEST_CASE("Testing Data Ringbuffer implementation", "[data-handling]") {
  rafko_mainframe::RafkoSettings settings;
  std::uint32_t buffer_number = 5;
  std::uint32_t buffer_size = 30;
  std::vector<double> data_sample(buffer_size, (0.0));
  std::vector<double> previous_data_sample(buffer_size, (0.0));
  rafko_utilities::DataRingbuffer<> buffer(
      buffer_number, [buffer_size](std::vector<double> &element) {
        element = std::vector<double>(buffer_size, 0.0);
      });

  REQUIRE(buffer.buffer_size() == buffer_size);
  REQUIRE(buffer.get_sequence_size() == buffer_number);

  /* By default every data should be 0 */
  for (std::uint32_t i = 0; i < buffer_number; ++i)
    check_data_match(data_sample, buffer.get_element(i));

  /* Adding numbers */
  for (std::uint32_t variant = 0; variant < (buffer_number * 2); ++variant) {
    check_data_match(data_sample, buffer.get_element(0));
    check_data_match(previous_data_sample, buffer.get_element(1));
    std::copy(data_sample.begin(), data_sample.end(),
              previous_data_sample.begin());
    buffer.copy_step();
    for (std::uint32_t b = 0; b < buffer_size; ++b) {
      data_sample[b] += b;
      buffer.get_element(0)[b] = data_sample[b];
    }
  }

  /* resetting buffers */
  buffer.reset();
  for (std::uint32_t past_index = 0u; past_index < buffer_number;
       ++past_index) {
    for (const double &number : buffer.get_element(past_index)) {
      REQUIRE(number == 0.0);
    }
  }
}

/*###############################################################################################
 * Testing a sequence of runs to be stored in the ringbuffer, and seeing if the
 * indexing is as expected by querying sequence indices and comparing to past
 * reaches Used interfaces:
 * - get_sequence_size
 * - get_sequence_index
 * */
TEST_CASE("Testing if ringbuffer past indexing logic is as expected",
          "[data-handling]") {
  rafko_mainframe::RafkoSettings settings;
  std::uint32_t sequence_number = 5;
  std::uint32_t buffer_size = 30;
  rafko_utilities::DataRingbuffer<> buffer(
      sequence_number, [buffer_size](std::vector<double> &element) {
        element = std::vector<double>(buffer_size, 0.0);
      });
  std::vector<double> data_sample(buffer_size);

  /* Simulate some runs: each element in the buffer shall have the value of it's
   * past value */
  for (std::int32_t i = sequence_number - 1; i >= 0; --i) {
    buffer.copy_step();
    for (double &sample_element : data_sample)
      sample_element = i;
    std::copy(data_sample.begin(), data_sample.end(),
              buffer.get_element(0).begin());
  }

  /*!Note: To understand Sequential indexes in the Data ringbuffer, this might
  help: for(std::int32_t i = sequence_number-1; i >= 0; --i) std::cout << "[" <<
  i << "]-"; std::cout << "past index (buffer conents also in this example)" <<
  std::endl; for(std::uint32_t i = 0; i < sequence_number; i++) std::cout << "["
  << i << "]-"; std::cout << "sequence index" << std::endl; */
}

} /* namespace rafko_utilities_test */

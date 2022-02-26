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
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_utilities/models/data_pool.h"

#include "test/test_utility.h"

namespace rafko_utilities_test {

/*###############################################################################################
 * Testing if data pool allocates data with the correct parameters
 * */
TEST_CASE("Data Pool parameters", "[data-handling][data-pool]"){
  std::uint32_t variants = 100;
  rafko_utilities::DataPool<double> data_pool;
  for(std::uint32_t i = 0; i < variants; ++i){
    std::uint32_t buffer_size = rand()%500;
    std::vector<double>& buffer = data_pool.reserve_buffer(buffer_size);
    REQUIRE( buffer.size() == buffer_size );
    data_pool.release_buffer(buffer);
  }
}

/*###############################################################################################
 * Testing if data pool is thread-safe
 * */
void use_buffer_thread(rafko_utilities::DataPool<double>& data_pool){
  std::uint32_t buffer_size = rand()%500;
  std::vector<double>& buffer = data_pool.reserve_buffer(buffer_size);
  std::vector<double> test_buffer = std::vector<double>(buffer_size);
  for(std::uint32_t i = 0; i < buffer_size; ++i){
    test_buffer[i] = rand()%1000;
    buffer[i] = test_buffer[i];
  }
  for(std::uint32_t i = 0; i < buffer_size; ++i){
    REQUIRE( test_buffer[i] == buffer[i] );
  }
  data_pool.release_buffer(buffer);
}

TEST_CASE("Data Pool multi-thread access", "[data-handling][data-pool][multi-threading]"){
  std::uint32_t variants = 10;
  rafko_utilities::DataPool<double> data_pool;
  for(std::uint32_t i = 0; i < variants; ++i){
    std::vector<std::thread> threads = std::vector<std::thread>();
    for(std::uint32_t j = 0; j < variants; ++j){
      threads.push_back(std::thread(&use_buffer_thread, std::ref(data_pool)));
    }
    while(0 < threads.size())
      if(threads.back().joinable()){
        threads.back().join();
        threads.pop_back();
      }
  }
}

} /* namespace rafko_utilities_test */

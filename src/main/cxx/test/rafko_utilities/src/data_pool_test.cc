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
#include "test/test_utility.h"
#include <vector>
#include <thread>

#include "rafko_mainframe/models/service_context.h"
#include "rafko_utilities/models/data_pool.h"

namespace sparse_net_library_test {

using std::vector;
using std::thread;

using rafko_utilities::DataPool;

/*###############################################################################################
 * Testing if data pool allocates data with the correct parameters
 * */
TEST_CASE("Data Pool parameters", "[data-handling][data-pool]"){
  uint32 variants = 100;
  DataPool<sdouble32> data_pool;
  for(uint32 i = 0; i < variants; ++i){
    uint32 buffer_size = rand()%500;
    vector<sdouble32>& buffer = data_pool.reserve_buffer(buffer_size);
    REQUIRE( buffer.size() == buffer_size );
    data_pool.release_buffer(buffer);
  }
}

/*###############################################################################################
 * Testing if data pool is thread-safe
 * */
void use_buffer_thread(DataPool<sdouble32>& data_pool){
  uint32 buffer_size = rand()%500;
  vector<sdouble32>& buffer = data_pool.reserve_buffer(buffer_size);
  vector<sdouble32> test_buffer = vector<sdouble32>(buffer_size);
  for(uint32 i = 0; i < buffer_size; ++i){
    test_buffer[i] = rand()%1000;
    buffer[i] = test_buffer[i];
  }
  for(uint32 i = 0; i < buffer_size; ++i){
    REQUIRE( test_buffer[i] == buffer[i] );
  }
  data_pool.release_buffer(buffer);
}

TEST_CASE("Data Pool multi-thread access", "[data-handling][data-pool][multi-threading]"){
  uint32 variants = 10;
  DataPool<sdouble32> data_pool;
  for(uint32 i = 0; i < variants; ++i){
    vector<thread> threads = vector<thread>();
    for(uint32 j = 0; j < variants; ++j){
      threads.push_back(thread(&use_buffer_thread, std::ref(data_pool)));
    }
    while(0 < threads.size())
      if(threads.back().joinable()){
        threads.back().join();
        threads.pop_back();
      }
  }
}

} /* namespace sparse_net_library_test */

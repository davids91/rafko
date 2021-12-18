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
#include <catch2/catch_approx.hpp>

#include "rafko_utilities/services/thread_group.h"

#include "test/test_utility.h"

namespace rafko_utilities_test {

using std::vector;
using std::lock_guard;

using rafko_net::ThreadGroup;

/*###############################################################################################
 * Testing if data pool is thread-safe
 * */
TEST_CASE("Thread Group generic use-case test", "[thread-group]"){
  const uint32 number_of_threads = 5;
  vector<sdouble32> test_buffer;
  sdouble32 expected;
  sdouble32 result = 0;
  std::mutex cout_mutex;

  ThreadGroup pool(number_of_threads);

  for(uint32 i = 0; i < 1000; ++i){
    test_buffer = vector<sdouble32>(rand()%500);
    std::for_each(test_buffer.begin(),test_buffer.end(),[](sdouble32& element){
      element = rand()%10;
    });
    expected = std::accumulate(test_buffer.begin(),test_buffer.end(), 0.0);
    result = 0;
    std::function<void(uint32)> fnc = [&](int thread_index){
      sdouble32 sum = 0;
      size_t length = (test_buffer.size() / number_of_threads) + 1u;
      size_t start = length * thread_index;
      length = std::min(length, (test_buffer.size() - start));
      if(start < test_buffer.size()) /* More threads could be available, than needed */
        for(size_t i = 0; i < length; ++i) sum += test_buffer[start + i];
      { /* accumulate the full results */
        lock_guard<std::mutex> my_lock(cout_mutex);
        result += sum;
      }
    };
    pool.start_and_block(fnc);
    REQUIRE( Catch::Approx(expected).margin(0.00000000000001) == result );
  }
}

} /* namespace rafko_utilities_test */

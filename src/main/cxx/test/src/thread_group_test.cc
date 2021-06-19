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

#include "sparse_net_library/services/thread_group.h"

namespace sparse_net_library_test {

using std::vector;
using std::tuple;
using std::lock_guard;

using sparse_net_library::ThreadGroup;

/*###############################################################################################
 * Testing if data pool is thread-safe
 * */
TEST_CASE("Thread Group generic use-case test", "[thread-group]"){
  const uint32 number_of_threads = 5;
  vector<sdouble32> test_buffer;
  sdouble32 expected;
  sdouble32 result = 0;
  std::mutex cout_mutex;

  ThreadGroup<vector<sdouble32>&> pool(number_of_threads,[&](tuple<vector<sdouble32>&>& inputs, int thread_index){
    sdouble32 sum = 0;
    vector<sdouble32>& used_buffer = std::get<vector<sdouble32>&>(inputs);
    size_t length = (used_buffer.size() / number_of_threads) + 1u;
    size_t start = length * thread_index;
    length = std::min(length, (used_buffer.size() - start));
    if(start < used_buffer.size()) /* More threads could be available, than needed */
      for(size_t i = 0; i < length; ++i) sum += used_buffer[start + i];
    { /* accumulate the full results */
      lock_guard<std::mutex> my_lock(cout_mutex);
      result += sum;
    }
  });

  for(uint32 i = 0; i < 1000; ++i){
    test_buffer = vector<sdouble32>(rand()%500);
    std::for_each(test_buffer.begin(),test_buffer.end(),[](sdouble32& element){
      element = rand()%10;
    });
    expected = std::accumulate(test_buffer.begin(),test_buffer.end(), 0.0);
    result = 0;
    auto tpl = std::forward_as_tuple(test_buffer);
    pool.start_and_block(tpl);
    REQUIRE( Approx(expected).margin(0.00000000000001) == result );
  }
}

} /* namespace sparse_net_library_test */

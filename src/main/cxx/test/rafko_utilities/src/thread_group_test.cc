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

#include <memory>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "rafko_utilities/services/thread_group.h"

#include "test/test_utility.h"

namespace rafko_utilities_test {

TEST_CASE("Thread Group generic use-case test", "[thread-group]"){
  const uint32 number_of_threads = 5;
  std::vector<sdouble32> test_buffer;
  sdouble32 expected;
  sdouble32 result = 0;
  std::mutex result_mutex;

  rafko_utilities::ThreadGroup pool(number_of_threads);

  for(uint32 variant = 0; variant < 10u; ++variant){
    test_buffer = std::vector<sdouble32>(rand()%50);
    std::for_each(test_buffer.begin(),test_buffer.end(),[](sdouble32& element){
      element = rand()%10;
    });
    expected = std::accumulate(test_buffer.begin(),test_buffer.end(), 0.0);
    result = 0;
    std::function<void(uint32)> fnc = [&](uint32 thread_index){
      sdouble32 sum = 0;
      size_t length = (test_buffer.size() / number_of_threads) + 1u;
      size_t start = length * thread_index;
      length = std::min(length, (test_buffer.size() - start));
      if(start < test_buffer.size()) /* More threads could be available, than needed */
        for(size_t i = 0; i < length; ++i) sum += test_buffer[start + i];
      { /* accumulate the full results */
        std::lock_guard<std::mutex> my_lock(result_mutex);
        result += sum;
      }
    };
    pool.start_and_block(fnc);
    REQUIRE( Catch::Approx(expected).margin(0.00000000000001) == result );
  }
}

TEST_CASE("Testing if ThreadGroups can be combined", "[thread-group][multi-thread]"){
  const uint32 number_of_threads = 5;
  sdouble32 expected = double_literal(0.0);
  sdouble32 result = double_literal(0.0);
  std::mutex result_mutex;

  rafko_utilities::ThreadGroup outer_pool(number_of_threads);
  std::vector<std::unique_ptr<rafko_utilities::ThreadGroup>> inner_pools;
  for(uint32 thread_index = 0; thread_index < 10u; ++thread_index){
    inner_pools.push_back(std::make_unique<rafko_utilities::ThreadGroup>(number_of_threads));
  }

  for(uint32 variant = 0; variant < 10u; ++variant){
    expected = double_literal(0.0);
    sdouble32 check = 0;
    sdouble32 check2 = 0;
    std::vector<std::vector<sdouble32>> test_buffer(rand()%5 + 1, std::vector<sdouble32>(rand()%5 + 1));
    std::for_each(test_buffer.begin(),test_buffer.end(),[&](std::vector<sdouble32>& vec){
      std::for_each(vec.begin(),vec.end(),[&](sdouble32& element){
        element = rand()%10;
        expected += element;
        check2 += element;
      });
      sdouble32 tmp = std::accumulate(vec.begin(),vec.end(), 0.0);
      check += tmp;
      // std::cout << "Partial result: " << tmp << std::endl;
    });

    result = double_literal(0.0);
    std::function<void(uint32)> outer_sum = [&](uint32 thread_index){
      size_t outer_length = (test_buffer.size() / number_of_threads) + 1u;
      size_t outer_start = outer_length * thread_index;
      outer_length = std::min(outer_length, (test_buffer.size() - outer_start));

      if(outer_start < test_buffer.size()){ /* More threads could be available, than needed */
        sdouble32 partial_sum = 0;
        std::mutex partial_mutex;

        for(size_t i = 0; i < outer_length; ++i){
          uint32 outer_index = outer_start + i;
          std::function<void(uint32)> inner_sum = [&](uint32 inner_thread_index){
            uint32 vec_to_do = outer_index;
            size_t length = (test_buffer[vec_to_do].size() / number_of_threads) + 1u;
            size_t start = length * inner_thread_index;
            length = std::min(length, (test_buffer[vec_to_do].size() - start));

            if(start < test_buffer[vec_to_do].size()){
              sdouble32 micro_sum = double_literal(0.0);
              for(size_t i = 0; i < length; ++i)
                micro_sum += test_buffer[vec_to_do][start + i];
              { /* accumulate the partial results */
                std::lock_guard<std::mutex> my_lock(partial_mutex);
                partial_sum += micro_sum;
              }
            }
          };
          inner_pools[thread_index]->start_and_block(inner_sum);
        }

        { /* accumulate the full results */
          std::lock_guard<std::mutex> my_lock(result_mutex);
          result += partial_sum;
        }

      }
    };

    outer_pool.start_and_block(outer_sum);
    REQUIRE( Catch::Approx(expected).margin(0.00000000000001) == result );
  }
}

} /* namespace rafko_utilities_test */

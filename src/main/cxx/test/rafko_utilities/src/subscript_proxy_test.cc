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
#include <iterator>
#include <catch2/catch_test_macros.hpp>

#include "rafko_utilities/models/subscript_proxy.h"

#include "test/test_utility.h"

namespace rafko_utilities_test {

TEST_CASE("Testing if Subscript Proxy works as expected", "[data-handling][proxy]"){
  std::vector<double> base_vector(rand()%100 + 1);
  std::unordered_map<std::size_t, std::size_t> dictionary;
  double tmp_value = 0.0;
  std::transform(base_vector.begin(), base_vector.end(), base_vector.begin(),
  [&tmp_value](const double&){
    tmp_value += 1.0;
    return tmp_value - 1.0;
  });

  while(0.0 < tmp_value){
    std::size_t index = rand()%base_vector.size();
    while(0 != dictionary.count(index))index = rand()%base_vector.size();
    dictionary.insert({index, rand()%base_vector.size()});
    tmp_value -= static_cast<double>(rand()%10) / 2.0;
    if(dictionary.size() == base_vector.size())break;
  }

  rafko_utilities::SubscriptProxy<> proxy(
    base_vector, std::make_shared<std::unordered_map<std::size_t, std::size_t>>(dictionary)
  );
  for(std::uint32_t i = 0; i < base_vector.size(); ++i){
    if(0u == dictionary.count(i))
      REQUIRE( i == proxy[i] );
      else REQUIRE( dictionary[i] == proxy[i] );
  }
  for(std::uint32_t i = 0; i < base_vector.size(); ++i){
    proxy[i] = i;
    if(0u == dictionary.count(i))
      REQUIRE( base_vector[i] == i );
      else{
        auto found = dictionary.find(i);
        REQUIRE( base_vector[found->second] == i );
      }
  }/*for(i in base_vector)*/
}

} /* namespace rafko_utilities_test */

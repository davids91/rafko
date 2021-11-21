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
#include <iterator>

#include "rafko_utilities/models/const_vector_subrange.h"

namespace rafko_net_test {

using std::vector;

/*###############################################################################################
 * Testing if Subvector range works as expected
 * */
TEST_CASE("Vector subrange", "[data-handling][sub-range]"){
  std::vector<uint8> big_vec = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  rafko_utilities::ConstVectorSubrange<uint8> my_range = { big_vec.cbegin(), big_vec.cend() };

  REQUIRE( big_vec.size() == my_range.size() );
  CHECK( big_vec.back() == my_range.back() );

  for(uint32 i = 0; i < big_vec.size(); i++){
    CHECK( big_vec[i] == my_range[i] );
  }

  for(uint32 variant = 0; variant < 100u; variant++){
    uint16 start = rand()%big_vec.size();
    uint16 num = rand()%(big_vec.size() - start);

    rafko_utilities::ConstVectorSubrange<uint8> range = rafko_utilities::ConstVectorSubrange<uint8>(
      big_vec.begin() + start, num
    );

    for(int i = 0; i < num; i++){
      CHECK( big_vec[start + i] == range[i] );
    }
  }
}

} /* namespace rafko_net_test */

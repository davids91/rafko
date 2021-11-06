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

#include "rafko_net/services/weight_experience_space.h"

namespace rafko_net_test {

using rafko_net::WeightExperienceSpace;

/*###############################################################################################
 * Testing whether or not the correct weight values are generated in a weight experience space
 * */
TEST_CASE("Testing weight experience space weight values","[weightxp]"){
  sdouble32 weight_min = double_literal(-1.0);
  sdouble32 weight_max = double_literal(1.0);
  sdouble32 weight_step = double_literal(0.2);
  WeightExperienceSpace wxp_space = WeightExperienceSpace(weight_min, weight_max, weight_step);

  uint32 weight_index = 0;
  for(sdouble32 weight = weight_min; weight < weight_max; weight += weight_step){
    REQUIRE( wxp_space.get_weights().size() > weight_index );
    REQUIRE( wxp_space.get_weight_experiences().size() > weight_index );
    CHECK( weight == wxp_space.get_weight(weight_index) );
    ++weight_index;
  }
}

} /* namespace rafko_net_test */

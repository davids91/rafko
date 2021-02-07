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

#include "sparse_net_library/services/Weight_experience_space.h"

namespace sparse_net_library_test {

using sparse_net_library::Weight_experience_space;

/*###############################################################################################
 * Testing whether or not the correct weight values are generated in a weight experience space
 * */
TEST_CASE("Testing weight experience space weight values","[weightxp]"){
  sdouble32 weight_min = double_literal(0.0);
  sdouble32 weight_max = double_literal(1.0);
  sdouble32 weight_step = double_literal(0.2);
  uint32 number_of_weights_in_space = static_cast<uint32>((weight_max - weight_min)/weight_step);
  Weight_experience_space wxp_space = Weight_experience_space(weight_min,weight_max,weight_step);


  sdouble32 tmp_weight = weight_min;
  for(uint32 bad_xp_iterator = 1; bad_xp_iterator < number_of_weights_in_space; ++bad_xp_iterator){
    CHECK(tmp_weight == wxp_space.get_best_weight());
    CHECK((tmp_weight + weight_step) == wxp_space.add_experience(double_literal(-1.0)));
    tmp_weight += weight_step;
  }
}

} /* namespace sparse_net_library_test */
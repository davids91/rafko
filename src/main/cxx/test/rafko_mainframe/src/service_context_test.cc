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

#include "rafko_mainframe/models/rafko_settings.h"

#include "test/test_utility.h"

namespace rafko_mainframe_test {

TEST_CASE("Testing if service settings learning rate Schedule is providing the expected learning rates with learning rate decay", "[service]" ) {
  sdouble32 learning_rate = double_literal(10.0);
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings().set_learning_rate(learning_rate).set_learning_rate_decay({
    {5u,double_literal(0.5f)},{10u,double_literal(0.5f)},{15u,double_literal(0.5f)},{20u,double_literal(0.5f)},{25u,double_literal(0.5f)},
  });
  for(int iteration = 0; iteration < 30; ++iteration){
    if( (5 == iteration)||(10 == iteration)||(15 == iteration)||(20 == iteration)||(25 == iteration) )
    learning_rate *= double_literal(0.5);
    sdouble32 boi = settings.get_learning_rate(iteration);
    CHECK( learning_rate == boi );
  }
}

} /* namespace rafko_mainframe_test */

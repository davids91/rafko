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

#include "rafko_mainframe/services/rafko_assertion_logger.h"

#include "test/test_utility.h"

namespace rafko_mainframe_test {

TEST_CASE("Testing Asserion system", "[assert]" ) {
  std::uint16_t small_value = 5;
  RFASSERT( 4 <= 5 );
  RFASSERT( 5 == 5 );
  RFASSERT( 5 <= 6 );
  RFASSERT( 6u != small_value );
  RFASSERT( 5u == small_value );

  /* create new Scope */
  RFASSERT_SCOPE(test_scope);
  RFASSERT_LOG("Trying a message...");
  RFASSERT( false );
}

} /* namespace rafko_mainframe_test */

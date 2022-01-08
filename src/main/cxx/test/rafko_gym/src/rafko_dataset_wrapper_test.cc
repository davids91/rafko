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

#include <catch2/catch_test_macros.hpp>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_gym/models/rafko_dataset_wrapper.h"

#include "test/test_utility.h"

namespace rafko_gym_test {

/*###############################################################################################
 * Testing Dataset creation
 * */
TEST_CASE("Testing Dataset wrapper creation", "[data-handling]" ) {
  rafko_mainframe::RafkoSettings settings;
  uint32 sample_number = 50;
  uint32 sequence_size = 6;
  sdouble32 expected_label = double_literal(50.0);
  rafko_gym::DataSet data_set = rafko_test::create_dataset(1/* input size */,1/* feature size */,sample_number, sequence_size, expected_label);
  rafko_gym::RafkoDatasetWrapper data_agr(data_set);
  REQUIRE( 0 == data_agr.get_prefill_inputs_number() );
  REQUIRE( sample_number == data_agr.get_number_of_sequences() );
}

} /* namespace rafko_gym_test */

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
TEST_CASE("Testing Dataset wrapper creation", "[environment][data-handling]" ) {
  rafko_mainframe::RafkoSettings settings;
  srand(2511793749);
  for(uint32 variant = 0; variant < 10; ++variant){
    uint32 sample_number = (rand()%5) + 1;
    uint32 sequence_size = (rand()%2) + 1;
    uint32 feature_size = (rand()%5) + 1;
    sdouble32 expected_label = static_cast<sdouble32>(rand()%10) * double_literal(100.0);
    std::unique_ptr<rafko_gym::DataSet> dataset(rafko_test::create_dataset(1/* input size */, feature_size, sample_number, sequence_size, expected_label));
    rafko_gym::RafkoDatasetWrapper data_wrap(*dataset);
    REQUIRE( 0 == data_wrap.get_prefill_inputs_number() );
    REQUIRE( sample_number == data_wrap.get_number_of_sequences() );
    for(uint32 sequence_index = 0; sequence_index < sample_number; ++sequence_index){
      for(uint32 label_index = 0; label_index < sequence_size; ++label_index){
        for(uint32 feature_index = 0; feature_index < feature_size; ++feature_index){
          REQUIRE(
            dataset->labels((sequence_index * sequence_size) + (label_index * feature_size) + feature_index)
            == data_wrap.get_label_sample((sequence_index * sequence_size) + label_index)[feature_index]
          );
        }/*for(every feature element)*/
      }/*for(every raw_label)*/
    }/*for(every sequence)*/
  }/*for(10 variants)*/
}

} /* namespace rafko_gym_test */

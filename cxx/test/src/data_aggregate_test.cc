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
#include <memory>

#include "gen/common.pb.h"
#include "rafko_mainframe/models/service_context.h"
#include "sparse_net_library/models/data_aggregate.h"
#include "sparse_net_library/models/cost_function_mse.h"

namespace sparse_net_library_test {

using std::unique_ptr;

using sparse_net_library::Data_set;
using sparse_net_library::Data_aggregate;
using sparse_net_library::Cost_function_mse;
using rafko_mainframe::Service_context;

/*###############################################################################################
 * Testing Data aggregate implementation and seeing if it converts @Data_set correctly
 * into the data item wih statistics, and take care of statistic error data correctly
 * */
TEST_CASE("Testing Data aggregate for non-seuqeuntial data", "[data-handling]" ) {
  Service_context service_context;
  uint32 sample_number = 50;
  uint32 sequence_size = 5;
  sdouble32 expected_label = double_literal(50.0);
  sdouble32 set_distance = double_literal(10.0);

  /* Create a @Data_set and fill it with data */
  Data_set data_set = Data_set();
  data_set.set_input_size(1);
  data_set.set_feature_size(1);
  data_set.set_sequence_size(sequence_size);

  for(uint32 i = 0; i < (sample_number * sequence_size); ++i){
    data_set.add_inputs(expected_label); /* Input should be irrelevant here */
    data_set.add_labels(expected_label);
  }

  /* Create @Data_aggregate from @Data_set */
  Data_aggregate data_agr(data_set, std::make_unique<Cost_function_mse>(1, service_context));
  REQUIRE( 0 == data_agr.get_prefill_inputs_number() );
  REQUIRE( sample_number == data_agr.get_number_of_sequences() );

  /* Test statistics for it */
  CHECK(double_literal(1.0) == data_agr.get_error_sum() ); /* Initial error should be exactly 1.0 */
  sdouble32 error_sum = double_literal(0.0);
  for(uint32 i = 0; i < data_agr.get_number_of_label_samples(); ++i){
    error_sum += data_agr.get_error(i);
  }
  CHECK( Approx(error_sum).epsilon(0.00000000000001) == data_agr.get_error_sum() );

  /* Set all features to the given distance */
  for(uint32 i = 0; i < (sample_number * sequence_size); ++i){
    data_agr.set_feature_for_label(i,{expected_label - set_distance});
    REQUIRE( /* Error: (distance^2)/(2 * overall number of samples) */
      Approx(
        pow(set_distance,2)/(double_literal(2.0)*(sample_number * sequence_size))
      ).epsilon(0.00000000000001) == data_agr.get_error(i)
    );
  }

  CHECK( /* Error: (distance^2)/(2 * overall number of samples) */
    Approx(
      pow(set_distance,2)/double_literal(2.0)
    ).epsilon(0.00000000000001) == data_agr.get_error_sum()
  );

  /* test if setting to different labels correclty updates the error sum */
  sdouble32 previous_error = data_agr.get_error_sum();
  error_sum = previous_error;
  sdouble32 faulty_feature;
  uint32 label_index;
  for(uint32 variant = 0;variant < 100; ++variant){
    label_index = rand()%(data_agr.get_number_of_label_samples());
    previous_error = data_agr.get_error(label_index);
    faulty_feature = ( data_agr.get_label_sample(label_index)[0] + set_distance );
    error_sum = (
      error_sum - previous_error /* remove the current label from the sum */
      + (
        (pow((expected_label - faulty_feature),2)/(double_literal(2.0)*(sample_number * sequence_size)))
      ) /* and add the new error to it */
    );
    data_agr.set_feature_for_label(label_index, {faulty_feature});
    CHECK(
      data_agr.get_error(label_index)
      == (
        (pow((expected_label - faulty_feature),2)/(double_literal(2.0)*(sample_number * sequence_size)))
      )
    );
    REQUIRE( Approx(error_sum).epsilon(0.00000000000001) == data_agr.get_error_sum() );
  }
  CHECK( Approx(error_sum).epsilon(0.00000000000001) == data_agr.get_error_sum() );

  /* test if the error is stored correctly even when the data is provided in bulk */
  set_distance *= (rand()%10 / double_literal(10.0)); /* modify the set distance just to be sure */
  vector<vector<sdouble32>> neuron_data_simulation(((sample_number * sequence_size)/2), {(expected_label - set_distance)}); /* create dummy neuron data with the configured distance */
  for(uint32 variant = 0; variant < 100; ++variant){
    data_agr.set_features_for_labels(neuron_data_simulation, 0, 0, (sample_number * sequence_size)/2); /* set the error for the first half */
    data_agr.set_features_for_labels(neuron_data_simulation, 0, (sample_number * sequence_size)/2, (sample_number * sequence_size)/2); /* set the error for the second half */

    Catch::StringMaker<sdouble32>::precision  = 15;
    for(uint32 i = 0; i < (sample_number * sequence_size); ++i)
    REQUIRE( /* Error: (distance^2)/(2 * overall number of samples) */
      Approx(
        pow(set_distance,2)/(double_literal(2.0)*sample_number * sequence_size)
      ).epsilon(0.00000000000001) == data_agr.get_error(i)
    );

    REQUIRE( /* Error: (distance^2)/2 */
      Approx(pow(set_distance,2)/double_literal(2.0)).epsilon(0.00000000000001) == data_agr.get_error_sum()
    );
  }
}

} /* namespace sparse_net_library_test */

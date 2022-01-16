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
#include <catch2/catch_approx.hpp>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_gym/models/rafko_dataset_cost.h"
#include "rafko_gym/services/cost_function_mse.h"

#include "test/test_utility.h"

namespace rafko_gym_test {

/*###############################################################################################
 * Testing Data aggregate implementation and seeing if it converts @DataSet correctly
 * into the data item wih statistics, and take care of statistic error data correctly
 * */
TEST_CASE("Testing Data aggregate for sequential data", "[data-handling]" ) {
  rafko_mainframe::RafkoSettings settings;
  uint32 sample_number = 10;
  uint32 sequence_size = 6;
  uint32 raw_label_size = sample_number * sequence_size;
  sdouble32 expected_label = double_literal(50.0);
  sdouble32 set_distance = double_literal(10.0);

  std::unique_ptr<rafko_gym::DataSet> dataset(rafko_test::create_dataset(1/* input size */,1/* feature size */,sample_number, sequence_size, expected_label));

  /* Create @RafkoDatasetCost from @DataSet */
  rafko_gym::RafkoDatasetWrapper dataset_wrap(*dataset);
  rafko_gym::RafkoDatasetCost data_objective(settings, std::make_unique<rafko_gym::CostFunctionMSE>(settings));
  REQUIRE( 0 == dataset_wrap.get_prefill_inputs_number() );
  REQUIRE( sample_number == dataset_wrap.get_number_of_sequences() );

  /* Test initial error value, then a fully errorless state */
  for(uint32 i = 0; i < raw_label_size; ++i){
    CHECK( Catch::Approx(data_objective.set_feature_for_label(dataset_wrap, i,{expected_label})).margin(0.00000000000001) == double_literal(0.0) );
  }

  /* Set all features to the given distance */
  for(uint32 i = 0; i < raw_label_size; ++i){
    REQUIRE( /* Error: (distance^2)/(2 * overall number of samples) */
      Catch::Approx( pow(set_distance,2) / (double_literal(2.0) * raw_label_size) ).margin(0.00000000000001)
      == data_objective.set_feature_for_label(dataset_wrap, i,{expected_label - set_distance})
    );
  }

  /* test if the error is stored correctly even when the data is provided in bulk */
  std::vector<std::vector<sdouble32>> neuron_data_simulation; /* create dummy neuron data with the configured distance */
  /*!Note: since the simulated neuron data is always at the same generated value here, it doesn't matter where the evaluation starts from inside the neuron buffer,
   *       i.e. what is the value of neuron_buffer_index, as long as the evaluation is inside the bounds of the array.
   */
  for(uint32 variant = 0; variant < 10u; ++variant){
    set_distance *= ((rand()%10) / double_literal(10.0)) + 0.1f;
    neuron_data_simulation = std::vector<std::vector<sdouble32>>((raw_label_size/2), {(expected_label - set_distance)});

    /* Test if the half of the set can be updated in bulk */
    REQUIRE( /* Error: (distance^2)/(2 * overall number of samples) */
      Catch::Approx(pow(set_distance,2) / (double_literal(2.0) * raw_label_size) ).margin(0.00000000000001)
      == data_objective.set_features_for_labels(dataset_wrap, neuron_data_simulation, 0, 0, raw_label_size/2)
    );
    REQUIRE( /* Error: (distance^2)/(2 * overall number of samples) */
      Catch::Approx( pow(set_distance,2) / (double_literal(2.0) * raw_label_size) ).margin(0.00000000000001)
      == data_objective.set_features_for_labels(dataset_wrap, neuron_data_simulation, 0, raw_label_size/2, raw_label_size/2)
    );

    /* Test if the quarter of the set can be updated in bulk */
    REQUIRE( /* Error: (distance^2)/(2 * overall number of samples) */
      Catch::Approx( pow(set_distance,2) / (double_literal(2.0) * raw_label_size) ).margin(0.00000000000001)
      == data_objective.set_features_for_labels(dataset_wrap, neuron_data_simulation, 0, (raw_label_size / 4) * 0, raw_label_size/4)
    );
    REQUIRE( /* Error: (distance^2)/(2 * overall number of samples) */
      Catch::Approx( pow(set_distance,2) / (double_literal(2.0) * raw_label_size) ).margin(0.00000000000001)
      == data_objective.set_features_for_labels(dataset_wrap, neuron_data_simulation, 0, (raw_label_size / 4) * 1, raw_label_size/4)
    );
    REQUIRE( /* Error: (distance^2)/(2 * overall number of samples) */
      Catch::Approx( pow(set_distance,2) / (double_literal(2.0) * raw_label_size) ).margin(0.00000000000001)
      == data_objective.set_features_for_labels(dataset_wrap, neuron_data_simulation, 0, (raw_label_size / 4) * 2, raw_label_size/4)
    );
    REQUIRE( /* Error: (distance^2)/(2 * overall number of samples) */
      Catch::Approx( pow(set_distance,2) / (double_literal(2.0) * raw_label_size) ).margin(0.00000000000001)
      == data_objective.set_features_for_labels(dataset_wrap, neuron_data_simulation, 0, (raw_label_size / 4) * 3, raw_label_size/4)
    );

    /* Check also the bulk sequenced interface */
    set_distance *= ((rand()%10) / double_literal(10.0)) + 0.1f;
    neuron_data_simulation = std::vector<std::vector<sdouble32>>((raw_label_size/2), {(expected_label - set_distance)});

    REQUIRE( /* Error: (distance^2)/(2 * overall number of samples) */
      Catch::Approx( pow(set_distance,2) / (double_literal(2.0) * sample_number *sequence_size) ).margin(0.00000000000001)
      == data_objective.set_features_for_sequences(dataset_wrap, neuron_data_simulation, 0, (sample_number * 0)/2, (sample_number/2), 0, dataset_wrap.get_sequence_size())
    );

    REQUIRE( /* Error: (distance^2)/(2 * overall number of samples) */
      Catch::Approx( pow(set_distance,2) / (double_literal(2.0) * sample_number *sequence_size) ).margin(0.00000000000001)
      == data_objective.set_features_for_sequences(dataset_wrap, neuron_data_simulation, 0, (sample_number * 1)/2, (sample_number/2), 0, dataset_wrap.get_sequence_size())
    );

    /* Check also with sequence truncation */
    set_distance *= ((rand()%10) / double_literal(10.0)) + 0.1f;
    neuron_data_simulation = std::vector<std::vector<sdouble32>>((raw_label_size/2), {(expected_label - set_distance)});
    REQUIRE( /* Error: (distance^2)/(2 * overall number of samples) *//*!Note: Because the truncation removes half of the evaluations, the end result should be half as as well.. */
      Catch::Approx( pow(set_distance,2) / (double_literal(4.0) * raw_label_size) ).margin(0.00000000000001)
      == data_objective.set_features_for_sequences(dataset_wrap, neuron_data_simulation, 0, (sample_number * 1)/2, (sample_number/2), dataset_wrap.get_sequence_size()/2, dataset_wrap.get_sequence_size()/2)
    );

  }/*for(variants)*/
}

} /* namespace rafko_gym_test */

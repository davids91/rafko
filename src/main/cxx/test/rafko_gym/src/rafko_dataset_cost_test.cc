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
  uint32 sample_number = 50;
  uint32 sequence_size = 6;
  sdouble32 expected_label = double_literal(50.0);
  sdouble32 set_distance = double_literal(10.0);

  /* Create a @DataSet and fill it with data */
  rafko_gym::DataSet dataset = rafko_gym::DataSet();
  dataset.set_input_size(1);
  dataset.set_feature_size(1);
  dataset.set_sequence_size(sequence_size);

  for(uint32 i = 0; i < (sample_number * sequence_size); ++i){
    dataset.add_inputs(expected_label); /* Input should be irrelevant here */
    dataset.add_labels(expected_label);
  }

  /* Create @RafkoDatasetCost from @DataSet */
  rafko_gym::RafkoDatasetWrapper dataset_wrap(dataset);
  rafko_gym::RafkoDatasetCost data_agr(settings, dataset_wrap, std::make_unique<rafko_gym::CostFunctionMSE>(settings));
  REQUIRE( 0 == data_agr.get_dataset().get_prefill_inputs_number() );
  REQUIRE( sample_number == data_agr.get_dataset().get_number_of_sequences() );

  /* Test initial error value, then a fully errorless state */
  CHECK( Catch::Approx(data_agr.get_error_sum()).margin(0.00000000000001) == double_literal(1.0) );
  for(uint32 i = 0; i < (sample_number * sequence_size); ++i){
    data_agr.set_feature_for_label(i,{expected_label});
  }
  sdouble32 initial_error = data_agr.get_error_sum();
  CHECK( Catch::Approx(initial_error).margin(0.00000000000001) == double_literal(0.0) );

  /* Test statistics for it */
  sdouble32 error_sum = double_literal(0.0);
  for(uint32 i = 0; i < data_agr.get_dataset().get_number_of_label_samples(); ++i){
    error_sum += data_agr.get_error(i);
  }
  CHECK( Catch::Approx(error_sum).margin(0.00000000000001) == data_agr.get_error_sum() );

  /* Set all features to the given distance */
  for(uint32 i = 0; i < (sample_number * sequence_size); ++i){
    data_agr.set_feature_for_label(i,{expected_label - set_distance});
    REQUIRE( /* Error: (distance^2)/(2 * overall number of samples) */
      Catch::Approx(
        pow(set_distance,2) / (double_literal(2.0) * (sample_number * sequence_size))
      ).margin(0.00000000000001) == data_agr.get_error(i)
    );
  }

  CHECK( /* Error: (distance^2)/(2 * overall number of samples) */
    Catch::Approx(
      pow(set_distance,2) / double_literal(2.0)
    ).margin(0.00000000000001) == data_agr.get_error_sum()
  );

  /* test if setting to different labels correclty updates the error sum */
  sdouble32 previous_error = data_agr.get_error_sum();
  error_sum = previous_error;
  sdouble32 faulty_feature;
  uint32 label_index;
  for(uint32 variant = 0;variant < 100; ++variant){
    label_index = rand()%(data_agr.get_dataset().get_number_of_label_samples());
    previous_error = data_agr.get_error(label_index);
    faulty_feature = ( data_agr.get_dataset().get_label_sample(label_index)[0] + set_distance );
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
        (pow((expected_label - faulty_feature),2) / (double_literal(2.0) * (sample_number * sequence_size)))
      )
    );
    REQUIRE( Catch::Approx(error_sum).epsilon(0.00000000000001) == data_agr.get_error_sum() );
  }
  CHECK( Catch::Approx(error_sum).epsilon(0.00000000000001) == data_agr.get_error_sum() );

  /* test if the error is stored correctly even when the data is provided in bulk */
  std::vector<std::vector<sdouble32>> neuron_data_simulation; /* create dummy neuron data with the configured distance */
  /*!Note: since the simulated neuron data is always at the same generated value here, it doesn't matter where the evaluation starts from inside the neuron buffer,
   *       i.e. what is the value of neuron_buffer_index, as long as the evaluation is inside the bounds of the array.
   */
  for(uint32 variant = 0; variant < 100; ++variant){
    set_distance *= ((rand()%10) / double_literal(10.0)) + 0.1f;
    neuron_data_simulation = std::vector<std::vector<sdouble32>>(((sample_number * sequence_size)/2), {(expected_label - set_distance)});

    /* Test if the half of the set can be updated in bulk */
    data_agr.set_features_for_labels(neuron_data_simulation, 0, 0, (sample_number * sequence_size)/2); /* set the error for the first half */
    data_agr.set_features_for_labels(neuron_data_simulation, 0, (sample_number * sequence_size)/2, (sample_number * sequence_size)/2); /* set the error for the second half */

    for(uint32 i = 0; i < (sample_number * sequence_size); ++i)
    REQUIRE( /* Error: (distance^2)/(2 * overall number of samples) */
      Catch::Approx(
        pow(set_distance,2) / (double_literal(2.0) * sample_number * sequence_size)
      ).margin(0.00000000000001) == data_agr.get_error(i)
    );

    REQUIRE( /* Error: (distance^2)/2 */
      Catch::Approx(pow(set_distance,2)/double_literal(2.0)).margin(0.00000000000001) == data_agr.get_error_sum()
    );

    /* Test if the quarter of the set can be updated in bulk */
    data_agr.set_features_for_labels(neuron_data_simulation, 0, (sample_number * sequence_size * 0)/4, (sample_number * sequence_size)/4);
    data_agr.set_features_for_labels(neuron_data_simulation, 0, (sample_number * sequence_size * 1)/4, (sample_number * sequence_size)/4);
    data_agr.set_features_for_labels(neuron_data_simulation, 0, (sample_number * sequence_size * 2)/4, (sample_number * sequence_size)/4);
    data_agr.set_features_for_labels(neuron_data_simulation, 0, (sample_number * sequence_size * 3)/4, (sample_number * sequence_size)/4);

    for(uint32 i = 0; i < (sample_number * sequence_size); ++i)
    REQUIRE( /* Error: (distance^2)/(2 * overall number of samples) */
      Catch::Approx(
        pow(set_distance,2) / (double_literal(2.0) * sample_number * sequence_size)
      ).margin(0.00000000000001) == data_agr.get_error(i)
    );

    REQUIRE( /* Error: (distance^2)/2 */
      Catch::Approx(pow(set_distance,2)/double_literal(2.0)).margin(0.00000000000001) == data_agr.get_error_sum()
    );

    /* Check also the bulk sequenced interface */
    set_distance *= ((rand()%10) / double_literal(10.0)) + 0.1f;
    neuron_data_simulation = std::vector<std::vector<sdouble32>>(((sample_number * sequence_size)/2), {(expected_label - set_distance)});
    data_agr.set_features_for_sequences(neuron_data_simulation, 0, (sample_number * 0)/2, (sample_number/2), 0, data_agr.get_dataset().get_sequence_size());
    data_agr.set_features_for_sequences(neuron_data_simulation, 0, (sample_number * 1)/2, (sample_number/2), 0, data_agr.get_dataset().get_sequence_size());

    for(uint32 i = 0; i < (sample_number * sequence_size); ++i)
    REQUIRE( /* Error: (distance^2)/(2 * overall number of samples) */
      Catch::Approx(
        pow(set_distance,2) / (double_literal(2.0) * sample_number *sequence_size)
      ).margin(0.00000000000001) == data_agr.get_error(i)
    );

    REQUIRE( /* Error: (distance^2)/2 */
      Catch::Approx(pow(set_distance,2)/double_literal(2.0)).margin(0.00000000000001) == data_agr.get_error_sum()
    );

    /* Check also with sequence truncation */
    sdouble32 old_set_distence = set_distance;
    set_distance *= ((rand()%10) / double_literal(10.0)) + 0.1f;
    neuron_data_simulation = std::vector<std::vector<sdouble32>>(((sample_number * sequence_size)/2), {(expected_label - set_distance)});
    data_agr.set_features_for_sequences(neuron_data_simulation, 0, (sample_number * 0)/2, (sample_number/2), data_agr.get_dataset().get_sequence_size()/2, data_agr.get_dataset().get_sequence_size()/2);
    data_agr.set_features_for_sequences(neuron_data_simulation, 0, (sample_number * 1)/2, (sample_number/2), data_agr.get_dataset().get_sequence_size()/2, data_agr.get_dataset().get_sequence_size()/2);

    uint32 raw_label_index = 0;
    for(uint32 sequence_index = 0; sequence_index < sample_number; ++sequence_index){
      for(uint32 sequence_iterator = 0; sequence_iterator < data_agr.get_dataset().get_sequence_size(); ++sequence_iterator){
        if((data_agr.get_dataset().get_sequence_size()/2) > sequence_iterator){ /* first half of the sequence should have have the new error */
          REQUIRE( /* Error: (distance^2)/(2 * overall number of samples) */
            Catch::Approx(
              pow(old_set_distence,2) / (double_literal(2.0) * sample_number * sequence_size)
            ).margin(0.00000000000001) == data_agr.get_error(raw_label_index)
          );
        }else{ /* second half of the sequence should have the new error */
          REQUIRE( /* Error: (distance^2)/(2 * overall number of samples) */
            Catch::Approx(
              pow(set_distance,2) / (double_literal(2.0) * sample_number * sequence_size)
            ).margin(0.00000000000001) == data_agr.get_error(raw_label_index)
          );
        }
        ++raw_label_index;
      }
    }

  }/*for(variants)*/
}

/*###############################################################################################
 * Testing if state changes inside the data aggregate persist, and push/pop operations
 * are working as expected
 * */
TEST_CASE("Testing Data aggregate for state changes", "[data-handling]" ) {
  rafko_mainframe::RafkoSettings settings;
  const uint32 sample_number = 50;
  const uint32 sequence_size = 5;
  const uint32 selected_index = rand()%(sample_number * sequence_size);
  const sdouble32 expected_label = double_literal(50.0);
  const sdouble32 set_distance = double_literal(10.0);
  sdouble32 initial_error;

  /* Create a @DataSet and fill it with data */
  rafko_gym::DataSet dataset = rafko_gym::DataSet();
  dataset.set_input_size(1);
  dataset.set_feature_size(1);
  dataset.set_sequence_size(sequence_size);

  for(uint32 i = 0; i < (sample_number * sequence_size); ++i){
    dataset.add_inputs(expected_label); /* Input should be irrelevant here */
    dataset.add_labels(expected_label);
  }

  rafko_gym::RafkoDatasetWrapper dataset_wrap(dataset);
  rafko_gym::RafkoDatasetCost data_agr(settings, dataset_wrap, std::make_unique<rafko_gym::CostFunctionMSE>(settings));
  REQUIRE( 0 == data_agr.get_dataset().get_prefill_inputs_number() );
  REQUIRE( sample_number == data_agr.get_dataset().get_number_of_sequences() );

  for(uint32 i = 0; i < (sample_number * sequence_size); ++i){
    data_agr.set_feature_for_label(i,{expected_label});
  }

  /* Saving state, modifying a feature */
  initial_error = data_agr.get_error_sum();
  CHECK( Catch::Approx(double_literal(0.0)).margin(0.00000000000001) == data_agr.get_error(selected_index) );
  data_agr.push_state();
  data_agr.set_feature_for_label(selected_index,{(expected_label - set_distance)});
  CHECK( Catch::Approx(double_literal(0.0)).margin(0.00000000000001) != data_agr.get_error(selected_index) );
  CHECK(
    Catch::Approx( /* Error: (distance^2)/(2 * overall number of samples) */
      pow(set_distance,2)/(double_literal(2.0) * sample_number * sequence_size)
    ).margin(0.00000000000001) == data_agr.get_error(selected_index)
  );
  CHECK( Catch::Approx(initial_error).margin(0.00000000000001) != data_agr.get_error_sum() );

  /* Restoring state, the error should be the same */
  data_agr.pop_state();
  CHECK( Catch::Approx(initial_error).margin(0.00000000000001) == data_agr.get_error_sum() );
  CHECK( Catch::Approx(double_literal(0.0)).margin(0.00000000000001) == data_agr.get_error(selected_index) );
}

} /* namespace rafko_gym_test */

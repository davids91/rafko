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
#include <memory>
#include <functional>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/solution.pb.h"
#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_utilities/models/data_ringbuffer.h"
#include "rafko_net/models/cost_function_mse.h"
#include "rafko_gym/models/data_aggregate.h"
#include "rafko_gym/services/rafko_environment_data_set.h"

#include "test/test_utility.h"

namespace rafko_gym_test {

/*###############################################################################################
 * Testing if the data-set environment produces correct error values
 * */
class DummyRafkoAgent : public rafko_gym::RafkoAgent{
public:
  DummyRafkoAgent(rafko_net::Solution& solution) : RafkoAgent(solution,0,0,4) { }
  void solve(const std::vector<sdouble32>&, rafko_utilities::DataRingbuffer& output, const std::vector<std::reference_wrapper<std::vector<sdouble32>>>&, uint32, uint32 ) const{
    output = result;
  }
  void set_result(sdouble32 value){
    result.set_element(0,0,value);
  }
  ~DummyRafkoAgent() = default;
private:
  rafko_utilities::DataRingbuffer result{1,1};
};

TEST_CASE("Testing Dataset environment", "[environment]"){
  uint32 sample_number = 50;
  uint32 sequence_size = 6;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_minibatch_size(10);
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

  /* Create the environment and dummy agent */
  rafko_gym::RafkoDatasetWrapper dataset_wrap(dataset);
  rafko_gym::DataAggregate training_cost(settings, dataset_wrap, std::make_unique<rafko_net::CostFunctionMSE>(settings));
  rafko_gym::RafkoEnvironmentDataSet environment(settings, dataset, dataset, rafko_net::cost_function_mse);
  rafko_net::Solution solution;
  solution.set_neuron_number(1);
  solution.set_output_neuron_number(1);
  solution.set_network_memory_length(1);
  solution.set_network_input_size(1);
  solution.add_cols(1);
  DummyRafkoAgent agent(solution);
  environment.install_agent(agent);

  /* Set some error and see if the environment produces the expected */
  agent.set_result(expected_label - set_distance);
  for(uint32 feature_index = 0; feature_index < dataset_wrap.get_number_of_label_samples(); ++feature_index)
    training_cost.set_feature_for_label( feature_index, {expected_label - set_distance} );
  sdouble32 environment_error = environment.full_evaluation();
  REQUIRE( /* One Error: (distance^2)/(2 * overall number of samples) */
    Catch::Approx( /* Error sum: One Error * overall number of samples  */
      pow(set_distance,2) / double_literal(2.0)
    ).margin(0.00000000000001) == -environment_error
  );

  /* Set another error stochastically; see if the error remains the same */
  set_distance *= static_cast<sdouble32>((rand()%10) + 1) / double(10.0);
  uint32 seed = rand();

  srand(seed);
  uint32 sequence_start_index = (rand()%(training_cost.get_dataset().get_number_of_sequences() - settings.get_minibatch_size() + 1));
  uint32 start_index_inside_sequence = (rand()%( /* If the memory is truncated for the training.. */
    training_cost.get_dataset().get_sequence_size() - settings.get_memory_truncation() + 1 /* ..not all result output values are evaluated.. */
  )); /* ..only settings.get_memory_truncation(), starting at a random index inside bounds */

  for(uint32 sequence_index = sequence_start_index; sequence_index < (sequence_start_index + settings.get_minibatch_size()); ++sequence_index){
    for(uint32 label_index = 0; label_index < settings.get_memory_truncation(); ++label_index){
      training_cost.set_feature_for_label(
        ((sequence_index * training_cost.get_dataset().get_sequence_size()) + start_index_inside_sequence + label_index),
        {expected_label - set_distance}
      );
    }
  }
  sdouble32 reference_error = -training_cost.get_error_sum();
  agent.set_result(expected_label - set_distance);
  sdouble32 measured_error = environment.stochastic_evaluation(seed);
  CHECK( Catch::Approx(reference_error).margin(0.00000000000001) == measured_error );
}

} /* namespace rako_gym_test */

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
#include "rafko_mainframe/models/rafko_service_context.h"
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
  rafko_mainframe::RafkoServiceContext service_context = rafko_mainframe::RafkoServiceContext()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_minibatch_size(10);
  sdouble32 expected_label = double_literal(50.0);
  sdouble32 set_distance = double_literal(10.0);

  /* Create a @DataSet and fill it with data */
  rafko_gym::DataSet data_set = rafko_gym::DataSet();
  data_set.set_input_size(1);
  data_set.set_feature_size(1);
  data_set.set_sequence_size(sequence_size);

  for(uint32 i = 0; i < (sample_number * sequence_size); ++i){
    data_set.add_inputs(expected_label); /* Input should be irrelevant here */
    data_set.add_labels(expected_label);
  }

  /* Create the environment and dummy agent */
  rafko_gym::DataAggregate training_set(service_context, data_set, std::make_unique<rafko_net::CostFunctionMSE>(service_context));
  rafko_gym::DataAggregate test_set(service_context, data_set, std::make_unique<rafko_net::CostFunctionMSE>(service_context));
  rafko_gym::RafkoEnvironmentDataSet environment(service_context, training_set, test_set);
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
  uint32 sequence_start_index = (rand()%(training_set.get_number_of_sequences() - service_context.get_minibatch_size() + 1));
  training_set.push_state();
  for(uint32 sequence_index = sequence_start_index; sequence_index < (sequence_start_index + service_context.get_minibatch_size()); ++sequence_index){
    for(uint32 label_index = 0; label_index < training_set.get_sequence_size(); ++label_index){
      training_set.set_feature_for_label(
        ((sequence_index * training_set.get_sequence_size()) + label_index),
        {expected_label - set_distance}
      );
    }
  }
  sdouble32 reference_error = -training_set.get_error_sum();
  training_set.pop_state();
  agent.set_result(expected_label - set_distance);
  sdouble32 measured_error = environment.stochastic_evaluation(seed);
  CHECK( Catch::Approx(reference_error).margin(0.00000000000001) == measured_error );
}

} /* namespace rako_gym_test */

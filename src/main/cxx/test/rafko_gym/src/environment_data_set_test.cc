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
#include <functional>

#include "rafko_protocol/common.pb.h"
#include "rafko_protocol/solution.pb.h"
#include "rafko_mainframe/models/service_context.h"
#include "rafko_utilities/models/data_ringbuffer.h"
#include "rafko_net/models/cost_function_mse.h"
#include "rafko_gym/models/data_aggregate.h"
#include "rafko_gym/services/environment_data_set.h"

namespace rako_gym_test {

using std::vector;
using std::reference_wrapper;

using rafko_mainframe::Service_context;
using rafko_utilities::DataRingbuffer;
using rafko_gym::Agent;
using rafko_gym::Data_aggregate;
using rafko_gym::Environment_data_set;
using rafko_net::Solution;
using rafko_net::DataSet;
using rafko_net::Cost_function_mse;

/*###############################################################################################
 * Testing if the data-set environment produces correct error values
 * */
class DummyAgent : public Agent{
public:
  DummyAgent(Solution& solution) : Agent(solution,0,0,4) { }
  void solve(const vector<sdouble32>&, DataRingbuffer& output, const vector<reference_wrapper<vector<sdouble32>>>&, uint32 ) const{
    output = result;
  }
  void set_result(sdouble32 value){
    result.set_element(0,0,value);
  }
  ~DummyAgent() = default;
private:
  DataRingbuffer result{1,1};
};

TEST_CASE("Testing Dataset environment", "[environment]"){
  uint32 sample_number = 50;
  uint32 sequence_size = 6;
  Service_context service_context = Service_context()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_minibatch_size(10);
  sdouble32 expected_label = double_literal(50.0);
  sdouble32 set_distance = double_literal(10.0);

  /* Create a @DataSet and fill it with data */
  DataSet data_set = DataSet();
  data_set.set_input_size(1);
  data_set.set_feature_size(1);
  data_set.set_sequence_size(sequence_size);

  for(uint32 i = 0; i < (sample_number * sequence_size); ++i){
    data_set.add_inputs(expected_label); /* Input should be irrelevant here */
    data_set.add_labels(expected_label);
  }

  /* Create the environment and dummy agent */
  Data_aggregate training_set(service_context, data_set, std::make_unique<Cost_function_mse>(1, service_context));
  Data_aggregate test_set(service_context, data_set, std::make_unique<Cost_function_mse>(1, service_context));
  Environment_data_set environment(service_context, training_set, test_set);
  Solution solution;
  solution.set_neuron_number(1);
  solution.set_output_neuron_number(1);
  solution.set_network_memory_length(1);
  solution.add_cols(1);
  DummyAgent agent(solution);

  /* Set some error and see if the environment produces the expected */
  agent.set_result(expected_label - set_distance);
  sdouble32 environment_error = environment.full_evaluation(agent);
  REQUIRE( /* One Error: (distance^2)/(2 * overall number of samples) */
    Approx( /* Error sum: One Error * overall number of samples  */
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
  sdouble32 measured_error = environment.stochastic_evaluation(agent, seed);
  CHECK( Approx(reference_error).margin(0.00000000000001) == measured_error );
}

} /* namespace rako_gym_test */

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

#include "rafko_protocol/common.pb.h"
#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/models/rafko_service_context.h"
#include "rafko_utilities/models/const_vector_subrange.h"
#include "rafko_net/models/cost_function_mse.h"
#include "rafko_net/services/rafko_net_builder.h"
#include "rafko_net/services/solution_builder.h"
#include "rafko_net/services/function_factory.h"
#include "rafko_gym/models/data_aggregate.h"
#include "rafko_gym/services/rafko_environment_data_set.h"
#include "rafko_gym/services/rafko_net_approximizer.h"

namespace rafko_net_test {

using std::vector;
using std::cout;
using std::endl;
using std::flush;
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

using rafko_mainframe::RafkoServiceContext;
using rafko_net::RafkoNet;
using rafko_net::RafkoNetBuilder;
using rafko_net::CostFunctionMSE;
using rafko_net::cost_function_squared_error;
using rafko_net::transfer_function_identity;
using rafko_net::transfer_function_selu;
using rafko_net::transfer_function_relu;
using rafko_net::transfer_function_sigmoid;
using rafko_net::weight_updater_default;
using rafko_net::weight_updater_momentum;
using rafko_net::weight_updater_nesterovs;
using rafko_net::weight_updater_adam;
using rafko_net::weight_updater_amsgrad;
using rafko_net::FunctionFactory;
using rafko_net::Solution;
using rafko_net::SolutionBuilder;
using rafko_net::SolutionSolver;
using rafko_gym::RafkoEnvironment;
using rafko_gym::RafkoEnvironmentDataSet;
using rafko_gym::RafkoNetApproximizer;
using rafko_gym::DataAggregate;

/*###############################################################################################
 * Testing if the gradients are added to the fragment correctly
 * */
TEST_CASE("Testing aprroximization fragment handling","[approximize][fragments]"){
  google::protobuf::Arena arena;
  RafkoServiceContext service_context = RafkoServiceContext()
    .set_max_processing_threads(7)
    .set_learning_rate(1e-1).set_arena_ptr(&arena);

  /* Create nets */
  vector<RafkoNet*> nets = vector<RafkoNet*>();
  /*!Note: no need for smart pointers, because ownership is in the arena.
    * The builder automatically uses the arena pointer provided in the context.
    */
  nets.push_back(RafkoNetBuilder(service_context)
    .input_size(2).expected_input_range(double_literal(1.0))
    .allowed_transfer_functions_by_layer(
      {
        {transfer_function_selu}
      }
    ).dense_layers({1})
  );

  /* Create dataset, test set and aprroximizer */
  std::pair<vector<vector<sdouble32>>,vector<vector<sdouble32>>> tmp1 = create_addition_dataset(5);
  DataAggregate* train_set = google::protobuf::Arena::Create<DataAggregate>(
    service_context.get_arena_ptr(), service_context,
    vector<vector<sdouble32>>(std::get<0>(tmp1)),
    vector<vector<sdouble32>>(std::get<1>(tmp1)),
    cost_function_squared_error
  );
  tmp1 = create_addition_dataset(10);
  DataAggregate* test_set = google::protobuf::Arena::Create<DataAggregate>(
    service_context.get_arena_ptr(), service_context,
    vector<vector<sdouble32>>(std::get<0>(tmp1)),
    vector<vector<sdouble32>>(std::get<1>(tmp1)),
    cost_function_squared_error
  );
  RafkoEnvironmentDataSet env(service_context, *train_set, *test_set);
  RafkoNetApproximizer approximizer(service_context, *nets[0], env, weight_updater_default);

  /* adding a simple-weight-gradient fragment */
  uint32 weight_index = rand()%(nets[0]->weight_table_size());
  uint32 gradient_value_index;
  sdouble32 weight_gradient = double_literal(0.5);
  sdouble32 weight_old_value = nets[0]->weight_table(weight_index);

  REQUIRE( nets[0]->weight_table(weight_index) == weight_old_value );

  approximizer.add_to_fragment(weight_index, weight_gradient);
  CHECK( 1 == approximizer.get_fragment().values_size() );
  CHECK( 1 == approximizer.get_fragment().weight_synapses_size() );
  CHECK( weight_gradient == approximizer.get_fragment().values(0) );
  gradient_value_index = approximizer.get_fragment().weight_synapses(0).starts();
  REQUIRE( static_cast<sint32>(gradient_value_index) < nets[0]->weight_table_size() );

  approximizer.apply_fragment(); /* Add the negative gradient */
  REQUIRE(
    (nets[0]->weight_table(weight_index) + (weight_gradient * service_context.get_learning_rate()))
    == Approx(weight_old_value).epsilon(0.00000000000001)
  );

  /* Continously adding gradients into a single fragment, while redundantly collecting them to see that the effect is the same */
  vector<sdouble32> correct_weight_delta(nets[0]->weight_table_size(), double_literal(0.0));
  vector<sdouble32> initial_weights = {nets[0]->weight_table().begin(),nets[0]->weight_table().end()};
  for(uint32 variant = 0; variant < 10; ++variant){
    weight_index = rand()%(nets[0]->weight_table_size());
    weight_gradient = double_literal(10.0)/static_cast<sdouble32>(rand()%10 + 1);
    correct_weight_delta[weight_index] += weight_gradient;
    approximizer.add_to_fragment(weight_index, weight_gradient);
  }
  for(weight_index = 0;static_cast<sint32>(weight_index) < nets[0]->weight_table_size(); ++weight_index){
    REQUIRE(
      nets[0]->weight_table(weight_index)
      == Approx(initial_weights[weight_index]).epsilon(0.00000000000001)
    );
  }
  approximizer.apply_fragment();
  for(weight_index = 0;static_cast<sint32>(weight_index) < nets[0]->weight_table_size(); ++weight_index){
    CHECK(
      Approx(nets[0]->weight_table(weight_index)).epsilon(0.00000000000001)
      == (initial_weights[weight_index] - (correct_weight_delta[weight_index] * service_context.get_learning_rate()))
    );
  }
}

/*###############################################################################################
 * Testing if the Sparse net library approximization convegres the network
 * - Generate dataset for addition
 *     - Input: 2 numbers between 0 and 1
 *     - Output: The summation of the two inputs
 * - Generate networks for datasets
 *     - 1 neuron
 *     - 1 layer
 *     - multi-layer
 * - For each dataset test if the each Net converges
 * */
TEST_CASE("Testing basic aprroximization","[approximize][feed-forward]"){
  google::protobuf::Arena arena;
  RafkoServiceContext service_context = RafkoServiceContext()
    .set_learning_rate(8e-2).set_minibatch_size(64).set_memory_truncation(2)
    .set_training_strategy(rafko_net::Training_strategy::training_strategy_stop_if_training_error_zero,true)
    .set_training_strategy(rafko_net::Training_strategy::training_strategy_early_stopping,false)
    .set_arena_ptr(&arena).set_max_solve_threads(2).set_max_processing_threads(4);
  uint32 number_of_samples = 128;

  /* Create nets */
  vector<RafkoNet*> nets = vector<RafkoNet*>();
  nets.push_back(RafkoNetBuilder(service_context)
    .input_size(2).expected_input_range(double_literal(1.0))
    .set_recurrence_to_self()
    .allowed_transfer_functions_by_layer({
      {transfer_function_selu}, {transfer_function_selu},
    }).dense_layers({10,1})
  );

  /* Create dataset, test set and optimizers; optimize nets */
  std::pair<vector<vector<sdouble32>>,vector<vector<sdouble32>>> tmp1 = create_sequenced_addition_dataset(number_of_samples, 4);
  DataAggregate* train_set =  google::protobuf::Arena::Create<DataAggregate>(
    service_context.get_arena_ptr(), service_context,
    vector<vector<sdouble32>>(std::get<0>(tmp1)),
    vector<vector<sdouble32>>(std::get<1>(tmp1)),
    cost_function_squared_error, /* Sequence size */4
  );
  tmp1 = create_sequenced_addition_dataset(number_of_samples * 2, 4);
  DataAggregate* test_set =  google::protobuf::Arena::Create<DataAggregate>(
    service_context.get_arena_ptr(), service_context,
    vector<vector<sdouble32>>(std::get<0>(tmp1)),
    vector<vector<sdouble32>>(std::get<1>(tmp1)),
    cost_function_squared_error, /* Sequence size */4
  );  RafkoEnvironmentDataSet env(service_context, *train_set, *test_set);
  RafkoNetApproximizer approximizer(service_context, *nets[0], env, weight_updater_amsgrad,1);

  sdouble32 train_error = 1.0;
  sdouble32 test_error = 1.0;
  sdouble32 minimum_error;
  uint32 number_of_steps;
  uint32 iteration;
  steady_clock::time_point start;
  uint32 average_duration;
  sdouble32 avg_gradient;

  train_error = 1.0;
  test_error = 1.0;
  number_of_steps = 0;
  average_duration = 0;
  iteration = 0;
  minimum_error = std::numeric_limits<sdouble32>::max();

  std::cout << "Approximizing net.." << std::endl;
  std::cout.precision(15);
  while(!approximizer.stop_training()){
    start = steady_clock::now();
    approximizer.collect_approximates_from_weight_gradients();
    avg_gradient = 0;
    for(sint32 frag_index = 0; frag_index < approximizer.get_weight_gradient().values_size(); ++frag_index){
      avg_gradient += approximizer.get_weight_gradient().values(frag_index);
    }
    avg_gradient /= static_cast<sdouble32>(approximizer.get_weight_gradient().values_size());

    approximizer.apply_fragment();
    auto current_duration = duration_cast<milliseconds>(steady_clock::now() - start).count();
    average_duration += current_duration;
    ++number_of_steps;
    train_error = train_set->get_error_avg();
    test_error = test_set->get_error_avg();
    if(abs(test_error) < minimum_error)minimum_error = abs(test_error);
    cout << "\tError:" << std::setprecision(9)
    << "Training:[" << train_error << "]; "
    << "Test:[" << test_error << "]; "
    << "Minimum: ["<< minimum_error <<"];"
    << "Avg_gradient: [" << avg_gradient << "]; "
    << "Iteration: ["<< iteration <<"];   "
    << "Duration: ["<< current_duration <<"ms];   "
    << std::endl;
    if(0 == (iteration % 100)){
      approximizer.full_evaluation();
      print_training_sample((rand()%number_of_samples), *train_set, *nets[0], service_context);
    }
    ++iteration;
  }
  if(1 < number_of_steps)average_duration /= number_of_steps;
  std::cout << std::endl << "Optimum reached in " << number_of_steps
  << " steps!(average runtime: "<< average_duration << " ms)" << endl;

  unique_ptr<SolutionSolver> after_solver(SolutionSolver::Builder(*SolutionBuilder(service_context).build(*nets[0]), service_context).build());

  sdouble32 error_summary[3] = {0,0,0};
  CostFunctionMSE after_cost(service_context);
  for(uint32 i = 0; i < number_of_samples; ++i){
    bool reset = 0 == (i%(train_set->get_sequence_size()));
    rafko_utilities::ConstVectorSubrange<> neuron_data = after_solver->solve(test_set->get_input_sample(i), reset);
    error_summary[0] += after_cost.get_feature_error({neuron_data.begin(),neuron_data.end()}, test_set->get_label_sample(i), number_of_samples);
  }
  std::cout << "==================================\n Error summaries:"
  << "\t"  << error_summary[0]
  << std::endl;

}

} /* namespace rafko_net_test */

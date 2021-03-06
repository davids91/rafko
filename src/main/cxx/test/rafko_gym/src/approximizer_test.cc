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

#include "gen/common.pb.h"
#include "gen/sparse_net.pb.h"
#include "rafko_mainframe/models/service_context.h"
#include "rafko_utilities/models/data_ringbuffer.h"
#include "sparse_net_library/models/cost_function_mse.h"
#include "sparse_net_library/services/sparse_net_builder.h"
#include "sparse_net_library/services/solution_builder.h"
#include "sparse_net_library/services/function_factory.h"
#include "rafko_gym/models/data_aggregate.h"
#include "rafko_gym/services/environment_data_set.h"
#include "rafko_gym/services/sparse_net_approximizer.h"

namespace sparse_net_library_test {

using std::vector;
using std::cout;
using std::endl;
using std::flush;
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

using rafko_mainframe::Service_context;
using rafko_utilities::DataRingbuffer;
using sparse_net_library::SparseNet;
using sparse_net_library::Sparse_net_builder;
using sparse_net_library::Cost_function_mse;
using sparse_net_library::COST_FUNCTION_SQUARED_ERROR;
using sparse_net_library::TRANSFER_FUNCTION_IDENTITY;
using sparse_net_library::TRANSFER_FUNCTION_SELU;
using sparse_net_library::TRANSFER_FUNCTION_RELU;
using sparse_net_library::TRANSFER_FUNCTION_SIGMOID;
using sparse_net_library::WEIGHT_UPDATER_DEFAULT;
using sparse_net_library::WEIGHT_UPDATER_MOMENTUM;
using sparse_net_library::WEIGHT_UPDATER_NESTEROV;
using sparse_net_library::WEIGHT_UPDATER_ADAM;
using sparse_net_library::WEIGHT_UPDATER_AMSGRAD;
using sparse_net_library::Function_factory;
using sparse_net_library::Solution;
using sparse_net_library::Solution_builder;
using sparse_net_library::Solution_solver;
using rafko_gym::Environment;
using rafko_gym::Environment_data_set;
using rafko_gym::Sparse_net_approximizer;
using rafko_gym::Data_aggregate;

/*###############################################################################################
 * Testing if the gradients are added to the fragment correctly
 * */
TEST_CASE("Testing aprroximization fragment handling","[approximize][fragments]"){
  google::protobuf::Arena arena;
  Service_context service_context = Service_context()
    .set_max_processing_threads(7)
    .set_step_size(1e-1).set_arena_ptr(&arena);

  /* Create nets */
  vector<SparseNet*> nets = vector<SparseNet*>();
  /*!Note: no need for smart pointers, because ownership is in the arena.
    * The builder automatically uses the arena pointer provided in the context.
    */
  nets.push_back(Sparse_net_builder(service_context)
    .input_size(2).expected_input_range(double_literal(1.0))
    .allowed_transfer_functions_by_layer(
      {
        {TRANSFER_FUNCTION_SELU}
      }
    ).dense_layers({1})
  );

  /* Create dataset, test set and aprroximizer */
  Data_aggregate* train_set = create_addition_dataset(5, *nets[0], COST_FUNCTION_SQUARED_ERROR, service_context);
  Data_aggregate* test_set = create_addition_dataset(5, *nets[0], COST_FUNCTION_SQUARED_ERROR, service_context);
  Environment_data_set env(service_context, *train_set, *test_set);
  Sparse_net_approximizer approximizer(service_context, *nets[0], env, WEIGHT_UPDATER_DEFAULT);

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
    (nets[0]->weight_table(weight_index) + (weight_gradient * service_context.get_step_size()))
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
      == (initial_weights[weight_index] - (correct_weight_delta[weight_index] * service_context.get_step_size()))
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
  Service_context service_context = Service_context()
    .set_step_size(2e-2).set_minibatch_size(64).set_memory_truncation(2)
    .set_arena_ptr(&arena).set_max_solve_threads(2).set_max_processing_threads(4);
  uint32 number_of_samples = 128;

  /* Create nets */
  vector<SparseNet*> nets = vector<SparseNet*>();
  nets.push_back(Sparse_net_builder(service_context)
    .input_size(2).expected_input_range(double_literal(1.0))
    .set_recurrence_to_layer()
    .allowed_transfer_functions_by_layer(
    //   {
    //     {TRANSFER_FUNCTION_SELU},
    //     {TRANSFER_FUNCTION_SELU},
    //     {TRANSFER_FUNCTION_SELU}
    //   }
    // ).dense_layers({10,10,1})
    {
      {TRANSFER_FUNCTION_SELU},
      {TRANSFER_FUNCTION_SELU},
    }
  ).dense_layers({2,1})

  );

  /* Create dataset, test set and optimizers; optimize nets */
  Data_aggregate* train_set = create_sequenced_addition_dataset(number_of_samples, 4, *nets[0], COST_FUNCTION_SQUARED_ERROR, service_context);
  Data_aggregate* test_set = create_sequenced_addition_dataset(number_of_samples * 2, 4, *nets[0], COST_FUNCTION_SQUARED_ERROR, service_context);
  Environment_data_set env(service_context, *train_set, *test_set);
  Sparse_net_approximizer approximizer(service_context, *nets[0], env, WEIGHT_UPDATER_AMSGRAD);

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
  while(abs(train_error) > service_context.get_step_size()){
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
    cout << "\rError:" << std::setprecision(9)
    << "Training:[" << train_error << "]; "
    << "Test:[" << test_error << "]; "
    << "Minimum: ["<< minimum_error <<"];"
    << "Avg_gradient: [" << avg_gradient << "]; "
    << "Iteration: ["<< iteration <<"];   "
    << "Duration: ["<< current_duration <<"ms];   "
    << std::endl;
    if(0 == (iteration % 100))
      print_training_sample((rand()%number_of_samples), *train_set, *nets[0], service_context);
    ++iteration;
  }
  average_duration /= number_of_steps;
  cout << endl << "Optimum reached in " << number_of_steps
  << " steps!(average runtime: "<< average_duration << " ms)" << endl;

  unique_ptr<Solution_solver> after_solver(Solution_solver::Builder(*Solution_builder(service_context).build(*nets[0]), service_context).build());

  sdouble32 error_summary[3] = {0,0,0};
  Cost_function_mse after_cost(1, service_context);
  for(uint32 i = 0; i < number_of_samples; ++i){
    bool reset = 0 == (i%(train_set->get_sequence_size()));
    const DataRingbuffer& neuron_data = after_solver->solve(test_set->get_input_sample(i), reset);
    error_summary[0] += after_cost.get_feature_error(neuron_data.get_const_element(0), test_set->get_label_sample(i), number_of_samples);
  }
  std::cout << "==================================\n Error summaries:"
  << "\t"  << error_summary[0]
  << std::endl;

}

} /* namespace sparse_net_library_test */

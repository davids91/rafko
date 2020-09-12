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
#include "sparse_net_library/models/data_aggregate.h"
#include "sparse_net_library/models/cost_function_mse.h"
#include "sparse_net_library/services/sparse_net_builder.h"
#include "sparse_net_library/services/solution_builder.h"
#include "sparse_net_library/services/sparse_net_approximizer.h"
#include "sparse_net_library/services/function_factory.h"

namespace sparse_net_library_test {

using std::vector;
using std::cout;
using std::endl;
using std::flush;
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
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
using sparse_net_library::Sparse_net_approximizer;
using sparse_net_library::Data_aggregate;
using sparse_net_library::Function_factory;
using sparse_net_library::Solution;
using sparse_net_library::Solution_builder;
using sparse_net_library::Solution_solver;
using rafko_mainframe::Service_context;

/*###############################################################################################
 * Testing if the gradients are added to the fragment correctly
 * */
TEST_CASE("Testing aprroximization fragment handling","[approximize][fragments]"){
  google::protobuf::Arena arena;
  Service_context service_context = Service_context().set_step_size(1e-1).set_arena_ptr(&arena);

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
  Sparse_net_approximizer approximizer(*nets[0],*train_set,*test_set,WEIGHT_UPDATER_NESTEROV, service_context);

  /* adding a simple-weight-gradient fragment */
  uint32 weight_index = rand()%(nets[0]->weight_table_size());
  uint32 gradient_value_index;
  sdouble32 weight_gradient = 0.5f;
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
    .set_step_size(3e-3).set_minibatch_size(128).set_memory_truncation(2)
    .set_arena_ptr(&arena).set_max_solve_threads(8);
  uint32 number_of_samples = 500;

  /* Create nets */
  vector<SparseNet*> nets = vector<SparseNet*>();
  nets.push_back(Sparse_net_builder(service_context)
    .input_size(2).expected_input_range(double_literal(1.0))
    .set_recurrence_to_layer()
    .allowed_transfer_functions_by_layer(
      {
        {TRANSFER_FUNCTION_SELU},
        {TRANSFER_FUNCTION_SELU}
      }
    ).dense_layers({2,1})
  );

  /* Create dataset, test set and optimizers; optimize nets */
  Data_aggregate* train_set = create_sequenced_addition_dataset(number_of_samples, 4, *nets[0], COST_FUNCTION_SQUARED_ERROR, service_context);
  Data_aggregate* test_set = create_sequenced_addition_dataset(number_of_samples, 4, *nets[0], COST_FUNCTION_SQUARED_ERROR, service_context);

  sdouble32 train_error = 1.0;
  sdouble32 test_error = 1.0;
  sdouble32 minimum_error;
  uint32 number_of_steps;
  uint32 iteration;
  steady_clock::time_point start;
  uint32 average_duration;

  train_error = 1.0;
  test_error = 1.0;
  number_of_steps = 0;
  average_duration = 0;
  iteration = 0;
  minimum_error = std::numeric_limits<sdouble32>::max();
  Sparse_net_approximizer approximizer(
    *nets[0], *train_set, *test_set, WEIGHT_UPDATER_NESTEROV, service_context
  );

  std::cout << "Optimizing net.." << std::endl;
  while(abs(train_error) > service_context.get_step_size()){
    start = steady_clock::now();
    approximizer.collect_approximates_from_random_direction();
    // if(0 == (iteration % 5))
      approximizer.apply_fragment();
    average_duration += duration_cast<milliseconds>(steady_clock::now() - start).count();
    ++number_of_steps;
    train_error = approximizer.get_train_error();
    test_error = approximizer.get_test_error();
    if(abs(test_error) < minimum_error)minimum_error = abs(test_error);
    cout << "\rError:"
    <<" training:[" << train_error << "]; "
    <<" test:[" << test_error << "]; "
    << "Minimum: ["<< minimum_error <<"]"
    << "Iteration: ["<< iteration <<"];                                           "
    << flush;

    ++iteration;
    if( /* every x iteration, until the step size if big enough to matter */
      (0 == (iteration % 2000))
      &&((service_context.get_epsilon() * 1000.0) < service_context.get_step_size())
    )service_context.set_step_size(service_context.get_step_size() * service_context.get_gamma()); /* make the step size decay */
  }
  average_duration /= number_of_steps;
  cout << endl << "Optimum reached in " << number_of_steps
  << " steps!(average runtime: "<< average_duration << " ms)" << endl;

  Solution_solver after_solver(*Solution_builder(service_context).build(*nets[0]), service_context);

  sdouble32 error_summary[3] = {0,0,0};
  Cost_function_mse after_cost(1, service_context);
  for(uint32 i = 0; i < number_of_samples; ++i){
    after_solver.solve(test_set->get_input_sample(i));
    error_summary[0] += after_cost.get_feature_error(after_solver.get_neuron_data(), test_set->get_label_sample(i), number_of_samples);
  }
  std::cout << "==================================\n Error summaries:"
  << "\t"  << error_summary[0]
  << std::endl;

}

} /* namespace sparse_net_library_test */
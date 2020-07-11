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
#include "sparse_net_global.h"
#include "gen/common.pb.h"
#include "gen/sparse_net.pb.h"
#include "rafko_mainframe/models/service_context.h"
#include "sparse_net_library/models/data_aggregate.h"
#include "sparse_net_library/models/cost_function_mse.h"
#include "sparse_net_library/services/sparse_net_builder.h"
#include "sparse_net_library/services/sparse_net_optimizer.h"
#include "sparse_net_library/services/function_factory.h"

#include <iostream>
#include <cstdlib>
#include <time.h>
#include <vector>
#include <float.h>
#include <cmath>
#include <memory>
#include <limits>
#include <chrono>
#include <string>
#include <iostream>
#include <fstream>

namespace sparse_net_library_test{

using std::unique_ptr;
using std::vector;
using std::cout;
using std::endl;
using std::flush;
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

using sparse_net_library::SparseNet;
using sparse_net_library::Sparse_net_builder;
using sparse_net_library::COST_FUNCTION_MSE;
using sparse_net_library::Sparse_net_optimizer;
using sparse_net_library::COST_FUNCTION_SQUARED_ERROR;
using sparse_net_library::TRANSFER_FUNCTION_IDENTITY;
using sparse_net_library::TRANSFER_FUNCTION_SELU;
using sparse_net_library::TRANSFER_FUNCTION_RELU;
using sparse_net_library::TRANSFER_FUNCTION_SIGMOID;
using sparse_net_library::WEIGHT_UPDATER_DEFAULT;
using sparse_net_library::WEIGHT_UPDATER_MOMENTUM;
using sparse_net_library::WEIGHT_UPDATER_NESTEROV;
using sparse_net_library::Data_aggregate;
using sparse_net_library::Function_factory;
using sparse_net_library::Solution;
using sparse_net_library::Solution_builder;
using sparse_net_library::Solution_solver;
using sparse_net_library::Cost_function_mse;
using rafko_mainframe::Service_context;

/*###############################################################################################
 * Testing if the Sparse net library optimization convegres the network
 * - Generate dataset for addition
 *     - Input: 2 numbers between 0 and 1
 *     - Output: The summation of the two inputs
 * - Generate networks for datasets
 *     - 1 neuron
 *     - 1 layer
 *     - multi-layer
 * - For each dataset test if the each Net converges
 * */
TEST_CASE("Testing basic optimization based on math","[optimize][feed-forward]"){
  uint32 number_of_samples = 500;

  vector<unique_ptr<SparseNet>> nets = vector<unique_ptr<SparseNet>>();
  nets.push_back(unique_ptr<SparseNet>(Sparse_net_builder()
    .input_size(2).expected_input_range(double_literal(1.0))
    .cost_function(COST_FUNCTION_SQUARED_ERROR)
    .allowed_transfer_functions_by_layer(
      {{TRANSFER_FUNCTION_SELU}}
    ).dense_layers({1})
  ));
  nets[0]->set_weight_table(1,0.9);
  nets[0]->set_weight_table(2,0.9);

  nets.push_back(unique_ptr<SparseNet>(Sparse_net_builder()
    .input_size(2).expected_input_range(double_literal(1.0))
    .cost_function(COST_FUNCTION_MSE)
    .allowed_transfer_functions_by_layer(
      {{TRANSFER_FUNCTION_SELU},{TRANSFER_FUNCTION_SELU}}
    ).dense_layers({2,1})
  ));
  nets[1]->set_weight_table(1,0.5);
  nets[1]->set_weight_table(2,0.5);
  nets[1]->set_weight_table(5,0.5);
  nets[1]->set_weight_table(6,0.5);
  nets[1]->set_weight_table(9,0.985);
  nets[1]->set_weight_table(10,0.985);

  nets.push_back(unique_ptr<SparseNet>(Sparse_net_builder()
    .input_size(2).expected_input_range(double_literal(1.0))
    .cost_function(COST_FUNCTION_MSE)
    .allowed_transfer_functions_by_layer(
      {{TRANSFER_FUNCTION_SELU},
       {TRANSFER_FUNCTION_SELU},
       {TRANSFER_FUNCTION_SELU}}
    ).dense_layers({2,2,1})
  ));

  nets[2]->set_weight_table(1,0.985);
  nets[2]->set_weight_table(2,0.985);
  nets[2]->set_weight_table(5,0.985);
  nets[2]->set_weight_table(6,0.985);
  nets[2]->set_weight_table(9,0.5);
  nets[2]->set_weight_table(10,0.5);
  nets[2]->set_weight_table(13,0.5);
  nets[2]->set_weight_table(14,0.5);
  nets[2]->set_weight_table(17,0.5);
  nets[2]->set_weight_table(18,0.5);

  /* Create data-set and test-set and optimize networks */
  Data_aggregate train_set = create_addition_dataset(number_of_samples,*nets[0]);

  Data_aggregate test_set = create_addition_dataset(number_of_samples,*nets[0]);

  sdouble32 train_error = 1.0;
  sdouble32 test_error = 1.0;
  sdouble32 minimum_error;
  uint32 number_of_steps;
  steady_clock::time_point start;
  uint32 average_duration;

  train_error = 1.0;
  test_error = 1.0;
  number_of_steps = 0;
  average_duration = 0;
  minimum_error = std::numeric_limits<sdouble32>::max();
  Sparse_net_optimizer optimizer(
    *nets[0],train_set,test_set,WEIGHT_UPDATER_DEFAULT,Service_context().set_step_size(1e-1)
  );

  std::cout << "Optimizing net.." << std::endl;
  while(abs(train_error) > 1e-1){
    start = steady_clock::now();
    optimizer.step();
    average_duration += duration_cast<milliseconds>(steady_clock::now() - start).count();
    ++number_of_steps;
    train_error = optimizer.get_train_error();
    test_error = optimizer.get_test_error();
    if(abs(test_error) < minimum_error)minimum_error = abs(test_error);
    cout << "\r Error:"
    <<" training:[" << train_error << "]; "
    <<" test:[" << test_error << "]; "
    << "Minimum: ["<< minimum_error <<"];                                           "
    << flush;
  }
  average_duration /= number_of_steps;
  cout << endl << "Optimum reached in " << number_of_steps
  << " steps!(average runtime: "<< average_duration << " ms)" << endl;

  Sparse_net_optimizer optimizer2(
    *nets[1], train_set, test_set, WEIGHT_UPDATER_MOMENTUM, Service_context().set_step_size(1e-1)
  ); /* .set_max_processing_threads(1)) for single-threaded tests */
  std::cout << "Optimizing bigger net.." << std::endl;
  train_set.reset_errors();
  test_set.reset_errors();
  train_error = 1.0;
  test_error = 1.0;
  number_of_steps = 0;
  minimum_error = std::numeric_limits<sdouble32>::max();
  while(abs(train_error) > 1e-1){
    start = steady_clock::now();
    optimizer2.step();
    average_duration += duration_cast<milliseconds>(steady_clock::now() - start).count();
    ++number_of_steps;
    train_error = optimizer2.get_train_error();
    test_error = optimizer2.get_test_error();
    if(abs(test_error) < minimum_error)minimum_error = abs(test_error);
    cout << "\r Error:"
    <<" training:[" << train_error << "]; "
    <<" test:[" << test_error << "]; "
    << "Minimum: ["<< minimum_error <<"];                                           "
    << flush;
  }
  average_duration /= number_of_steps;
  cout << endl << "Optimum reached in " << number_of_steps
  << " steps!(average runtime: "<< average_duration << " ms)" << endl;

  Sparse_net_optimizer optimizer3(
    *nets[2], train_set, test_set, WEIGHT_UPDATER_NESTEROV, Service_context().set_step_size(1e-1)
  );
  cout << "Optimizing biggest net.." << std::endl;
  train_set.reset_errors();
  test_set.reset_errors();
  train_error = 1.0;
  test_error = 1.0;
  number_of_steps = 0;
  minimum_error = std::numeric_limits<sdouble32>::max();
  while(abs(train_error) > 1e-1){
    start = steady_clock::now();
    optimizer3.step();
    average_duration += duration_cast<milliseconds>(steady_clock::now() - start).count();
    ++number_of_steps;
    train_error = optimizer3.get_train_error();
    test_error = optimizer3.get_test_error();
    if(abs(test_error) < minimum_error)minimum_error = abs(test_error);
    cout << "\r Error:"
    <<" training:[" << train_error << "]; "
    <<" test:[" << test_error << "]; "
    << "Minimum: ["<< minimum_error <<"];                                           "
    << flush;
  }
  average_duration /= number_of_steps;
  cout << endl << "Optimum reached in " << number_of_steps
  << " steps!(average runtime: "<< average_duration << " ms)" << endl;

  Solution_solver after_solver(*Solution_builder().build(*nets[0]));
  Solution_solver after_solver2(*Solution_builder().build(*nets[1]));
  Solution_solver after_solver3(*Solution_builder().build(*nets[2]));

  sdouble32 error_summary[3] = {0,0,0};
  Cost_function_mse after_cost(1,number_of_samples);
  for(uint32 i = 0; i < number_of_samples; ++i){
    after_solver.solve(test_set.get_input_sample(i));
    after_solver2.solve(test_set.get_input_sample(i));
    after_solver3.solve(test_set.get_input_sample(i));
    error_summary[0] += after_cost.get_feature_error(
      after_solver.get_neuron_data(), test_set.get_label_sample(i)
    );
    error_summary[1] +=after_cost.get_feature_error(
      after_solver2.get_neuron_data(), test_set.get_label_sample(i)
    );
    error_summary[2] += after_cost.get_feature_error(
      after_solver3.get_neuron_data(), test_set.get_label_sample(i)
    );
  }
  std::cout << "==================================\n Error summaries:"
  << "\t"  << error_summary[0]
  << "\t"  << error_summary[1]
  << "\t"  << error_summary[2]
  << std::endl;
}

/*###############################################################################################
 * Testing if the Sparse net library optimization can train networks for the binary addition
 * - Generate a dataset for binary addition:
 *     - Inputs: [0..1][0..1]
 *     - Outputs: [result][carry_bit]
 * - Generate networks for datasets
 *     - 2 neuron
 *     - multi-layer
 * - For each dataset test if the each Net converges
 * */
void print_training_sample(uint32 sample_sequence_index, Data_aggregate data_set, SparseNet& net){
    Solution_solver sample_solver(*Solution_builder().build(net));
    vector<sdouble32> neuron_data(data_set.get_sequence_size());
    std::cout.precision(2);
    std::cout << std::endl << "Training sample["<< sample_sequence_index <<"]:" << std::endl;
    for(uint32 j = 0;j < data_set.get_sequence_size();++j){
      std::cout << "["<< data_set.get_input_sample((data_set.get_sequence_size() * sample_sequence_index) + j)[0] <<"]";
    }
    std::cout << std::endl;
    for(uint32 j = 0;j < data_set.get_sequence_size();++j){
      std::cout << "["<< data_set.get_input_sample((data_set.get_sequence_size() * sample_sequence_index) + j)[1] <<"]";
    }
    std::cout << std::endl;
    std::cout << "--------------expected:" << std::endl;
    std::cout.precision(2);
    sample_solver.reset();
    for(uint32 j = 0;j < data_set.get_sequence_size();++j){
      std::cout << "["<< data_set.get_label_sample((data_set.get_sequence_size() * sample_sequence_index) + j)[0] <<"]";
      sample_solver.solve(data_set.get_input_sample((data_set.get_sequence_size() * sample_sequence_index) + j));
      neuron_data[j] = sample_solver.get_neuron_data().back();
    }
    std::cout << std::endl;
    std::cout << "------<>------actual:" << std::endl;

    for(uint32 j = 0;j < data_set.get_sequence_size();++j){
      std::cout << "["<< neuron_data[j] <<"]";
    }
    std::cout << std::endl;
    std::cout << "==============" << std::endl;
    std::cout.precision(15);
}

TEST_CASE("Testing recurrent Networks","[optimize][recurrent]"){
  uint32 sequence_size = 5;
  uint32 number_of_samples = 50;
  uint32 epoch = 100;

  /* Create nets */
  vector<unique_ptr<SparseNet>> nets = vector<unique_ptr<SparseNet>>();
  nets.push_back(unique_ptr<SparseNet>(Sparse_net_builder()
    .input_size(2).expected_input_range(double_literal(1.0))
    .set_recurrence_to_self()
    .cost_function(COST_FUNCTION_SQUARED_ERROR)
    .allowed_transfer_functions_by_layer(
      {
        {TRANSFER_FUNCTION_SELU},
        {TRANSFER_FUNCTION_SIGMOID}
      }
    ).dense_layers({5,1})
  ));

  /* Create dataset, test set and optimizers; optimize nets */
  Data_aggregate train_set = create_sequenced_addition_dataset(number_of_samples, sequence_size, *nets[0]);
  Data_aggregate test_set = create_sequenced_addition_dataset(number_of_samples, sequence_size, *nets[0]);

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
  Sparse_net_optimizer optimizer(
    *nets[0],train_set,test_set,WEIGHT_UPDATER_NESTEROV,Service_context().set_step_size(1e-2)
  );

  for(uint32 sample_sequence = 0; sample_sequence < number_of_samples; ++sample_sequence)
    print_training_sample(sample_sequence, train_set, *nets[0]);

  std::cout << "Optimizing net.." << std::endl;
  while(abs(train_error) > 1e-2){
    start = steady_clock::now();
    optimizer.step();
    average_duration += duration_cast<milliseconds>(steady_clock::now() - start).count();
    ++number_of_steps;
    train_error = optimizer.get_train_error();
    test_error = optimizer.get_test_error();
    if(abs(test_error) < minimum_error)minimum_error = abs(test_error);
    cout << "\r Error:"
    <<" training:[" << train_error << "]; "
    <<" test:[" << test_error << "]; "
    << "Minimum: ["<< minimum_error <<"]"
    << "Iteration: ["<< iteration <<"];                                           "
    << flush;
    ++iteration;
    if(0 == (iteration % epoch))
      print_training_sample((rand()%number_of_samples), train_set, *nets[0]);
    
  }
  average_duration /= number_of_steps;
  cout << endl << "Optimum reached in " << number_of_steps
  << " steps!(average runtime: "<< average_duration << " ms)" << endl;

  Solution_solver after_solver(*Solution_builder().build(*nets[0]));

  sdouble32 error_summary[3] = {0,0,0};
  Cost_function_mse after_cost(1,number_of_samples);
  for(uint32 i = 0; i < number_of_samples; ++i){
    after_solver.solve(test_set.get_input_sample(i));
    error_summary[0] += after_cost.get_feature_error(
      after_solver.get_neuron_data(), test_set.get_label_sample(i)
    );
  }
  std::cout << "==================================\n Error summaries:"
  << "\t"  << error_summary[0]
  << std::endl;
}

}/* namespace sparse_net_library_test */

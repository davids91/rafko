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

#include "sparse_net_global.h"
#include "gen/common.pb.h"
#include "gen/sparse_net.pb.h"
#include "models/service_context.h"
#include "models/data_aggregate.h"
#include "models/cost_function_mse.h"
#include "services/sparse_net_builder.h"
#include "services/sparse_net_optimizer.h"
#include "services/function_factory.h"

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
using sparse_net_library::uint32;
using sparse_net_library::sint32;
using sparse_net_library::sdouble32;
using sparse_net_library::SparseNet;
using sparse_net_library::Sparse_net_builder;
using sparse_net_library::COST_FUNCTION_MSE;
using sparse_net_library::Sparse_net_optimizer;
using sparse_net_library::COST_FUNCTION_SQUARED_ERROR;
using sparse_net_library::TRANSFER_FUNCTION_IDENTITY;
using sparse_net_library::TRANSFER_FUNCTION_SELU;
using sparse_net_library::TRANSFER_FUNCTION_SIGMOID;
using sparse_net_library::Service_context;
using sparse_net_library::WEIGHT_UPDATER_DEFAULT;
using sparse_net_library::WEIGHT_UPDATER_MOMENTUM;
using sparse_net_library::WEIGHT_UPDATER_NESTEROV;
using sparse_net_library::Data_aggregate;
using sparse_net_library::Function_factory;
using sparse_net_library::Solution;
using sparse_net_library::Solution_builder;
using sparse_net_library::Solution_solver;
using sparse_net_library::Cost_function_mse;

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
  vector<vector<sdouble32>> net_inputs_train(number_of_samples);
  vector<vector<sdouble32>> net_inputs_test(number_of_samples);
  vector<vector<sdouble32>> addition_dataset_train(number_of_samples);
  vector<vector<sdouble32>> addition_dataset_test(number_of_samples);

  srand(time(nullptr));
  sdouble32 max_x = DBL_MIN;
  sdouble32 max_y = DBL_MIN;
  for(uint32 i = 0;i < number_of_samples;++i){
    net_inputs_train[i].push_back(static_cast<sdouble32>(rand()%100));
    net_inputs_train[i].push_back(static_cast<sdouble32>(rand()%100));
    if(net_inputs_train[i][0] > max_x)max_x = net_inputs_train[i][0];
    if(net_inputs_train[i][1] > max_y)max_y = net_inputs_train[i][1];

    net_inputs_test[i].push_back(static_cast<sdouble32>(rand()%100));
    net_inputs_test[i].push_back(static_cast<sdouble32>(rand()%100));
    if(net_inputs_test[i][0] > max_x)max_x = net_inputs_test[i][0];
    if(net_inputs_test[i][1] > max_y)max_y = net_inputs_test[i][1];
  }

  for(uint32 i = 0;i < number_of_samples;++i){ /* Normalize the inputs */
    net_inputs_train[i][0] /= max_x;
    net_inputs_train[i][1] /= max_y;
    net_inputs_test[i][0] /= max_x;
    net_inputs_test[i][1] /= max_y;

    addition_dataset_train[i].push_back(net_inputs_train[i][0] + net_inputs_train[i][1]);
    addition_dataset_test[i].push_back(net_inputs_test[i][0] + net_inputs_test[i][1]);
  }

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
  Data_aggregate train_set(
    vector<vector<sdouble32>>(net_inputs_train),
    vector<vector<sdouble32>>(addition_dataset_train),
    *nets[0]
  );

  Data_aggregate test_set(
    vector<vector<sdouble32>>(net_inputs_test),
    vector<vector<sdouble32>>(addition_dataset_test),
    *nets[0]
  );

  sdouble32 train_error = 1.0;
  sdouble32 test_error = 1.0;
  sdouble32 minimum_error;
  uint32 number_of_steps;
  steady_clock::time_point start;
  uint32 average_duration;
  std::string file_name = "log_";
  file_name += std::to_string(std::chrono::system_clock::now().time_since_epoch().count() / 60);
  file_name += ".txt";
  std::cout.precision(15);

  //std::ofstream logfile;
  //logfile.open (file_name);

#if 1
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
    /*logfile << train_error << ",";
    for(int i = 0; i<nets[0]->weight_table_size(); ++i){
      logfile << nets[0]->weight_table(i) << ",";
    }
    logfile << "\n";*/
  }
  average_duration /= number_of_steps;
  cout << endl << "Optimum reached in " << number_of_steps
  << " steps!(average runtime: "<< average_duration << " ms)" << endl;
#endif
#if 1
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
    /*logfile << train_error << ",";
    for(int i = 0; i<nets[1]->weight_table_size(); ++i){
      logfile << nets[1]->weight_table(i) << ",";
    }
    logfile << "\n";*/
  }
  average_duration /= number_of_steps;
  cout << endl << "Optimum reached in " << number_of_steps
  << " steps!(average runtime: "<< average_duration << " ms)" << endl;
#endif
#if 1
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
    #if 1
    cout << "\r Error:"
    <<" training:[" << train_error << "]; "
    <<" test:[" << test_error << "]; "
    << "Minimum: ["<< minimum_error <<"];                                           "
    << flush;
    #endif
  }
  average_duration /= number_of_steps;
  cout << endl << "Optimum reached in " << number_of_steps
  << " steps!(average runtime: "<< average_duration << " ms)" << endl;
#endif
  Solution_solver after_solver(*Solution_builder().build(*nets[0]));
  Solution_solver after_solver2(*Solution_builder().build(*nets[1]));
  Solution_solver after_solver3(*Solution_builder().build(*nets[2]));

  sdouble32 error_summary[3] = {0,0,0};
  Cost_function_mse after_cost(1,number_of_samples);
  for(uint32 i = 0; i < number_of_samples; ++i){
    after_solver.solve(net_inputs_test[i]);
    after_solver2.solve(net_inputs_test[i]);
    after_solver3.solve(net_inputs_test[i]);
    error_summary[0] += after_cost.get_feature_error(
      after_solver.get_neuron_data(), addition_dataset_test[i]
    );
    error_summary[1] +=after_cost.get_feature_error(
      after_solver2.get_neuron_data(), addition_dataset_test[i]
    );
    error_summary[2] += after_cost.get_feature_error(
      after_solver3.get_neuron_data(), addition_dataset_test[i]
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
TEST_CASE("Testing recursive Networks","[optimize][recurrent]"){
  uint32 sequence_size = 5;
  uint32 number_of_samples = 500;
  uint32 carry_bit_train;
  uint32 carry_bit_test;
  vector<vector<sdouble32>> net_inputs_train(sequence_size * number_of_samples);
  vector<vector<sdouble32>> net_inputs_test(sequence_size * number_of_samples);
  vector<vector<sdouble32>> addition_dataset_train(sequence_size * number_of_samples);
  vector<vector<sdouble32>> addition_dataset_test(sequence_size * number_of_samples);

  for(uint32 i = 0;i < number_of_samples;++i){
    carry_bit_train = 0;
    carry_bit_test = 0;
    for(uint32 j = 0;j <sequence_size;++j){ /* Add testing and training sequences randomly */
      net_inputs_train[(sequence_size * i) + j] = vector<sdouble32>(2);
      net_inputs_test[(sequence_size * i) + j] = vector<sdouble32>(2);
      addition_dataset_train[(sequence_size * i) + j] = vector<sdouble32>(1);
      addition_dataset_test[(sequence_size * i) + j] = vector<sdouble32>(1);
      net_inputs_train[(sequence_size * i) + j][0] = static_cast<sdouble32>(rand()%2);
      net_inputs_train[(sequence_size * i) + j][1] = static_cast<sdouble32>(rand()%2);
      net_inputs_test[(sequence_size * i) + j][0] = static_cast<sdouble32>(rand()%2);
      net_inputs_test[(sequence_size * i) + j][1] = static_cast<sdouble32>(rand()%2);

      addition_dataset_train[(sequence_size * i) + j][0] =
        net_inputs_train[(sequence_size * i) + j][0]
        + net_inputs_train[(sequence_size * i) + j][1]
        + carry_bit_train;
      if(1 < addition_dataset_train[(sequence_size * i) + j][0]){
        addition_dataset_train[(sequence_size * i) + j][0] = 1;
        carry_bit_train = 1;
      }else{
        carry_bit_train = 0;
      }

      addition_dataset_test[(sequence_size * i) + j][0] =
        net_inputs_test[(sequence_size * i) + j][0]
        + net_inputs_test[(sequence_size * i) + j][1]
        + carry_bit_train;
      if(1 < addition_dataset_test[(sequence_size * i) + j][0]){
        addition_dataset_test[(sequence_size * i) + j][0] = 1;
        carry_bit_test = 1;
      }else{
        carry_bit_test = 0;
      }
    }
  }

  /* Print out the training data */
  std::cout << "==============" << std::endl;
  for(uint32 i = 0;i < number_of_samples;++i){
    for(uint32 j = 0;j < sequence_size;++j){
      std::cout << "["<< net_inputs_train[(sequence_size * i) + j][0] <<"]";
    }
    std::cout << std::endl;
    for(uint32 j = 0;j < sequence_size;++j){
      std::cout << "["<< net_inputs_train[(sequence_size * i) + j][1] <<"]";
    }
    std::cout << std::endl;
    std::cout << "--------------" << std::endl;
    for(uint32 j = 0;j < sequence_size;++j){
      std::cout << "["<< addition_dataset_train[(sequence_size * i) + j][0] <<"]";
    }
    std::cout << std::endl;
    std::cout << "==============" << std::endl;
  }/**/

  /* Create nets */
  vector<unique_ptr<SparseNet>> nets = vector<unique_ptr<SparseNet>>();
  nets.push_back(unique_ptr<SparseNet>(Sparse_net_builder()
    .input_size(2).expected_input_range(double_literal(1.0))
    .set_recurrence_to_layer()
    .cost_function(COST_FUNCTION_SQUARED_ERROR)
    .allowed_transfer_functions_by_layer(
      {
        {TRANSFER_FUNCTION_SELU},
        {TRANSFER_FUNCTION_SELU}
      }
    ).dense_layers({32,1})
  ));

  /* Create dataset, test set and optimizers; optimize nets */
  Data_aggregate train_set(
    vector<vector<sdouble32>>(net_inputs_train),
    vector<vector<sdouble32>>(addition_dataset_train),
    *nets[0], sequence_size
  );

  Data_aggregate test_set(
    vector<vector<sdouble32>>(net_inputs_test),
    vector<vector<sdouble32>>(addition_dataset_test),
    *nets[0], sequence_size
  );

  sdouble32 train_error = 1.0;
  sdouble32 test_error = 1.0;
  sdouble32 minimum_error;
  uint32 number_of_steps;
  steady_clock::time_point start;
  uint32 average_duration;
  std::string file_name = "log_";
  file_name += std::to_string(std::chrono::system_clock::now().time_since_epoch().count() / 60);
  file_name += ".txt";
  std::cout.precision(15);

  //std::ofstream logfile;
    //logfile.open (file_name);

#if 1
  train_error = 1.0;
  test_error = 1.0;
  number_of_steps = 0;
  average_duration = 0;
  minimum_error = std::numeric_limits<sdouble32>::max();
  Sparse_net_optimizer optimizer(
    *nets[0],train_set,test_set,WEIGHT_UPDATER_NESTEROV,Service_context().set_step_size(1e-2)
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
    /*logfile << train_error << ",";
    for(int i = 0; i<nets[0]->weight_table_size(); ++i){
      logfile << nets[0]->weight_table(i) << ",";
    }
    logfile << "\n";*/
  }
  average_duration /= number_of_steps;
  cout << endl << "Optimum reached in " << number_of_steps
  << " steps!(average runtime: "<< average_duration << " ms)" << endl;
#endif

  Solution_solver after_solver(*Solution_builder().build(*nets[0]));

  sdouble32 error_summary[3] = {0,0,0};
  Cost_function_mse after_cost(1,number_of_samples);
  for(uint32 i = 0; i < number_of_samples; ++i){
    after_solver.solve(net_inputs_test[i]);
    error_summary[0] += after_cost.get_feature_error(
      after_solver.get_neuron_data(), addition_dataset_test[i]
    );
  }
  std::cout << "==================================\n Error summaries:"
  << "\t"  << error_summary[0]
  << std::endl;
}

}/* namespace sparse_net_library_test */

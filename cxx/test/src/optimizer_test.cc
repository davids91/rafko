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
using sparse_net_library::sdouble32;
using sparse_net_library::SparseNet;
using sparse_net_library::Sparse_net_builder;
using sparse_net_library::COST_FUNCTION_MSE;
using sparse_net_library::TRANSFER_FUNCTION_IDENTITY;
using sparse_net_library::TRANSFER_FUNCTION_RELU;
using sparse_net_library::Sparse_net_optimizer;
using sparse_net_library::Service_context;
using sparse_net_library::WEIGHT_UPDATER_DEFAULT;
using sparse_net_library::WEIGHT_UPDATER_MOMENTUM;
using sparse_net_library::WEIGHT_UPDATER_NESTEROV;
using sparse_net_library::Data_aggregate;
using sparse_net_library::Function_factory;
using sparse_net_library::Solution_builder;
using sparse_net_library::Solution_solver;
using sparse_net_library::Cost_function_mse;

/*###############################################################################################
 * Testing if the Sparse net library optimization convegres the network
 * - Generate datasets
 *     - addition ( x + y )
 *     - subtraction ( x - y )
 *     - squared function ( x^2 )
 *     - squared fuction ( y^2 )
 * - Generate networks for datasets wherever the network would be adequate
 *     - 1 neuron
 *     - 1 layer
 *     - multi-layer
 * - For each dataset test if the each Net converges
 * */
TEST_CASE("Testing basic optimization based on math","[opt-test][opt-math]"){
  vector<vector<sdouble32>> net_inputs(500);
  vector<vector<sdouble32>> addition_dataset(500);
  vector<vector<sdouble32>> substraction_dataset(500);
  vector<vector<sdouble32>> square_x_dataset(500);
  vector<vector<sdouble32>> square_y_dataset(500);

  srand(time(nullptr));
  sdouble32 max_x = DBL_MIN;
  sdouble32 max_y = DBL_MIN;
  for(uint32 i = 0;i < 500;++i){
    net_inputs[i].push_back(static_cast<sdouble32>(rand()%100));
    net_inputs[i].push_back(static_cast<sdouble32>(rand()%100));
    if(net_inputs[i][0] > max_x)max_x = net_inputs[i][0];
    if(net_inputs[i][1] > max_y)max_y = net_inputs[i][1];
  }

  for(uint32 i = 0;i < 500;++i){ /* Normalize the inputs */
    net_inputs[i][0] /= max_x;
    net_inputs[i][1] /= max_y;

    addition_dataset[i].push_back(net_inputs[i][0] + net_inputs[i][1]);
    substraction_dataset[i].push_back(net_inputs[i][0] - net_inputs[i][1]);
    square_x_dataset[i].push_back(pow(net_inputs[i][0],2));
    square_y_dataset[i].push_back(pow(net_inputs[i][1],2));
  }

  vector<unique_ptr<SparseNet>> nets = vector<unique_ptr<SparseNet>>();
  nets.push_back(unique_ptr<SparseNet>(Sparse_net_builder()
    .input_size(2).expected_input_range(1.0)
    .cost_function(COST_FUNCTION_MSE)
    .allowed_transfer_functions_by_layer(
      {{TRANSFER_FUNCTION_RELU}}
    ).dense_layers({1})
  ));

  nets.push_back(unique_ptr<SparseNet>(Sparse_net_builder()
    .input_size(2).expected_input_range(1.0)
    .cost_function(COST_FUNCTION_MSE)
    .allowed_transfer_functions_by_layer(
      {{TRANSFER_FUNCTION_RELU},{TRANSFER_FUNCTION_RELU}}
    ).dense_layers({3,1})
  ));

  nets.push_back(unique_ptr<SparseNet>(Sparse_net_builder()
    .input_size(2).expected_input_range(1.0)
    .cost_function(COST_FUNCTION_MSE)
    .allowed_transfer_functions_by_layer(
      {{TRANSFER_FUNCTION_RELU},
       {TRANSFER_FUNCTION_RELU},
       {TRANSFER_FUNCTION_RELU}}
    ).dense_layers({3,2,1})
  ));

  Solution_solver solver(Solution_solver(*Solution_builder().build(*nets[0])));
  Solution_solver solver2(Solution_solver(*Solution_builder().build(*nets[1])));
  Solution_solver solver3(Solution_solver(*Solution_builder().build(*nets[2])));

  Data_aggregate data_aggregate(
    vector<vector<sdouble32>>(net_inputs),
    vector<vector<sdouble32>>(addition_dataset),
    *nets[0]
  );

  /* Optimize nets */
  sdouble32 last_error;
  sdouble32 minimum_error;
  uint32 number_of_steps;
  steady_clock::time_point start;
  uint32 average_duration;

  last_error = 5;
  number_of_steps = 0;
  average_duration = 0;
  minimum_error = std::numeric_limits<sdouble32>::max();
  Sparse_net_optimizer optimizer(
    *nets[0],data_aggregate,WEIGHT_UPDATER_MOMENTUM,Service_context().set_step_size(1e-1)
  );
  std::cout << "Optimizing net.." << std::endl;
  while(abs(last_error) > 1e-1){
    start = steady_clock::now();
    optimizer.step(50);
    average_duration += duration_cast<milliseconds>(steady_clock::now() - start).count();
    ++number_of_steps;
    last_error = optimizer.get_last_error();
    if(abs(last_error) < minimum_error)minimum_error = abs(last_error);
    cout << "\r Error: [" << last_error << "]; "
    << "Minimum: ["<< minimum_error <<"];                                           "
    << flush;
  }
  average_duration /= number_of_steps;
  cout << endl << "Optimum reached in " << number_of_steps
  << " steps!(average runtime: "<< average_duration << " ms)" << endl;

  Sparse_net_optimizer optimizer2(
    *nets[1],data_aggregate,WEIGHT_UPDATER_DEFAULT,Service_context().set_step_size(1e-2)
  ); /* .set_max_processing_threads(1)) for single-threaded tests */
  std::cout << "Optimizing bigger net.." << std::endl;
  data_aggregate.reset();
  last_error = 5;
  number_of_steps = 0;
  minimum_error = std::numeric_limits<sdouble32>::max();
  while(abs(last_error) > 1e-2){
    start = steady_clock::now();
    optimizer2.step(50);
    average_duration += duration_cast<milliseconds>(steady_clock::now() - start).count();
    ++number_of_steps;
    last_error = optimizer2.get_last_error();
    if(abs(last_error) < minimum_error)minimum_error = abs(last_error);
    cout << "\r Error: [" << last_error << "]; "
    << "Minimum: ["<< minimum_error <<"];                                           "
    << flush;
  }
  average_duration /= number_of_steps;
  cout << endl << "Optimum reached in " << number_of_steps
  << " steps!(average runtime: "<< average_duration << " ms)" << endl;

  Sparse_net_optimizer optimizer3(
    *nets[2],data_aggregate,WEIGHT_UPDATER_NESTEROV,Service_context().set_step_size(1e-6)
  );
  cout << "Optimizing biggest net.." << std::endl;
  data_aggregate.reset();
  last_error = 5;
  number_of_steps = 0;
  minimum_error = std::numeric_limits<sdouble32>::max();
  while(abs(last_error) > 1e-6){
    start = steady_clock::now();
    optimizer3.step(100);
    average_duration += duration_cast<milliseconds>(steady_clock::now() - start).count();
    ++number_of_steps;
    last_error = optimizer3.get_last_error();
    if(abs(last_error) < minimum_error)minimum_error = abs(last_error);
    cout << "\r Error: [" << last_error << "]; "
    << "Minimum: ["<< minimum_error <<"];                                           "
    << flush;
  }
  average_duration /= number_of_steps;
  cout << endl << "Optimum reached in " << number_of_steps
  << " steps!(average runtime: "<< average_duration << " ms)" << endl;
  Solution_solver after_solver(Solution_solver(*Solution_builder().build(*nets[0])));
  Solution_solver after_solver2(Solution_solver(*Solution_builder().build(*nets[1])));
  Solution_solver after_solver3(Solution_solver(*Solution_builder().build(*nets[2])));

  sdouble32 error_summary[3] = {0,0,0};
  Cost_function_mse after_cost(1,500);
  for(uint32 i = 0; i < 500; ++i){
    after_solver.solve(net_inputs[i]);
    after_solver2.solve(net_inputs[i]);
    after_solver3.solve(net_inputs[i]);
    error_summary[0] += after_cost.get_error(
      after_solver.get_neuron_data(0), addition_dataset[i][0]
    );
    error_summary[1] +=after_cost.get_error(
      after_solver2.get_neuron_data(3), addition_dataset[i][0]
    );
    error_summary[2] += after_cost.get_error(
      after_solver3.get_neuron_data(5), addition_dataset[i][0]
    );
  }
  std::cout << "Error summaries:" 
  << "\t"  << error_summary[0]
  << "\t"  << error_summary[1]
  << "\t"  << error_summary[2]
  << std::endl;

}

}/* namespace sparse_net_library_test */

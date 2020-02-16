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
 *    along with Foobar.  If not, see <https://www.gnu.org/licenses/> or
 *    <https://github.com/davids91/rafko/blob/master/LICENSE>
 */

#include "test/catch.hpp"

#include "sparse_net_global.h"
#include "gen/common.pb.h"
#include "gen/sparse_net.pb.h"
#include "services/sparse_net_builder.h"
#include "services/sparse_net_optimizer.h"

#include <iostream>
#include <cstdlib>
#include <time.h>
#include <vector>
#include <float.h>
#include <cmath>
#include <memory>
#include <limits>

namespace sparse_net_library_test{

using std::unique_ptr;
using std::vector;
using std::cout;
using std::endl;
using sparse_net_library::uint32;
using sparse_net_library::sdouble32;
using sparse_net_library::SparseNet;
using sparse_net_library::Sparse_net_builder;
using sparse_net_library::COST_FUNCTION_QUADRATIC;
using sparse_net_library::TRANSFER_FUNCTION_IDENTITY;
using sparse_net_library::Sparse_net_optimizer;

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
    .cost_function(COST_FUNCTION_QUADRATIC)
    .allowed_transfer_functions_by_layer(
      {{TRANSFER_FUNCTION_IDENTITY}}
    ).dense_layers({1})
  ));

  nets.push_back(unique_ptr<SparseNet>(Sparse_net_builder()
    .input_size(2).expected_input_range(1.0)
    .cost_function(COST_FUNCTION_QUADRATIC)
    .allowed_transfer_functions_by_layer(
      {{TRANSFER_FUNCTION_IDENTITY},{TRANSFER_FUNCTION_IDENTITY}}
    ).dense_layers({3,1})
  ));

  nets.push_back(unique_ptr<SparseNet>(Sparse_net_builder()
    .input_size(2).expected_input_range(1.0)
    .cost_function(COST_FUNCTION_QUADRATIC)
    .allowed_transfer_functions_by_layer(
      {{TRANSFER_FUNCTION_IDENTITY},
       {TRANSFER_FUNCTION_IDENTITY},
       {TRANSFER_FUNCTION_IDENTITY}}
    ).dense_layers({3,2,1})
  ));

  /* Optimize net */
  sdouble32 last_error;
  sdouble32 minimum_error = std::numeric_limits<sdouble32>::max();
  sdouble32 moving_average_error = 0;
  last_error = 5;
  Sparse_net_optimizer optimizer(*nets[0],addition_dataset);
  std::cout << "Optimizing net.." << std::endl;
  while(abs(last_error) > 1e-2){
    optimizer.step(net_inputs, 50, 1e-2);
    last_error = optimizer.get_last_error();
    if(abs(last_error) < minimum_error)minimum_error = abs(last_error);
    moving_average_error = (moving_average_error + last_error)/2;
    cout << "\r Error: [" << last_error << "]; "
    << "Average: ["<< moving_average_error <<"]; "
    << "Minimum: ["<< minimum_error <<"]; "
    << "                    " << std::flush;
  }
  cout << endl;

  Sparse_net_optimizer optimizer2(*nets[1],addition_dataset); /* Add sparse_net_library::Service_context().set_max_processing_threads(1)) for single-threaded tests */
  std::cout << "Optimizing bigger net.." << std::endl;
  last_error = 5;
  while(abs(last_error) > 1e-5){
    optimizer2.step(net_inputs, 50, 1e-5);
    last_error = optimizer2.get_last_error();
    if(abs(last_error) < minimum_error)minimum_error = abs(last_error);
    moving_average_error = (moving_average_error + last_error)/2;
    cout << "\r Error: [" << last_error << "]; "
    << "Minimum: ["<< minimum_error <<"]; "
    << "Average: ["<< moving_average_error <<"]; "
    << "                    " << std::flush;
  }
  cout << std::endl;

  Sparse_net_optimizer optimizer3(*nets[1],addition_dataset);
  std::cout << "Optimizing biggest net.." << std::endl;
  last_error = 5;
  while(abs(last_error) > 1e-4){
    optimizer3.step(net_inputs, 50, 1e-4);
    last_error = optimizer3.get_last_error();
    if(abs(last_error) < minimum_error)minimum_error = abs(last_error);
    moving_average_error = (moving_average_error + last_error)/2;
    cout << "\r Error: [" << last_error << "]; "
    << "Average: ["<< moving_average_error <<"]; "
    << "Minimum: ["<< minimum_error <<"]; "
    << "                    " << std::flush;
  }
  cout << endl;
}

}/* namespace sparse_net_library_test */

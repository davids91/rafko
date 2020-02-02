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

namespace sparse_net_library_test{

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

  srand(time(nullptr)); //srand(static_cast<uint32>(time(nullptr)));
  for(uint32 i = 0;i < 500;++i){
    net_inputs[i].push_back(static_cast<sdouble32>(rand()%100));
    net_inputs[i].push_back(static_cast<sdouble32>(rand()%100));
    addition_dataset[i].push_back(net_inputs[i][0] + net_inputs[i][1]);
    substraction_dataset[i].push_back(net_inputs[i][0] - net_inputs[i][1]);
    square_x_dataset[i].push_back(pow(net_inputs[i][0],2));
    square_y_dataset[i].push_back(pow(net_inputs[i][1],2));
  }

  vector<SparseNet> nets = vector<SparseNet>();
  nets.push_back(*Sparse_net_builder()
    .input_size(2).expected_input_range(100.0)
    .cost_function(COST_FUNCTION_QUADRATIC)
    .allowed_transfer_functions_by_layer(
      {{TRANSFER_FUNCTION_IDENTITY}}
    ).dense_layers({1})
  );

  nets.push_back(*Sparse_net_builder()
    .input_size(2).expected_input_range(100.0)
    .cost_function(COST_FUNCTION_QUADRATIC)
    .allowed_transfer_functions_by_layer(
      {{TRANSFER_FUNCTION_IDENTITY},{TRANSFER_FUNCTION_IDENTITY}}
    ).dense_layers({3,1})
  );

  nets.push_back(*Sparse_net_builder()
    .input_size(2).expected_input_range(100.0)
    .cost_function(COST_FUNCTION_QUADRATIC)
    .allowed_transfer_functions_by_layer(
      {{TRANSFER_FUNCTION_IDENTITY},
       {TRANSFER_FUNCTION_IDENTITY},
       {TRANSFER_FUNCTION_IDENTITY}}
    ).dense_layers({3,2,1})
  );

  /* Optimize net */
  Sparse_net_optimizer optimizer = Sparse_net_optimizer(nets[0],addition_dataset);
  std::cout << "Optimizing net.." << std::endl;
  while(true){
    optimizer.step(net_inputs,1e-10);
    cout << "\r\t\t Error: \t [" << optimizer.last_error() << "]\t\t";
  }
  cout << endl;
}

}/* namespace sparse_net_library_test */
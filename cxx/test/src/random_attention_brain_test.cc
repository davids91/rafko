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
#include "sparse_net_library/services/sparse_net_builder.h"
#include "sparse_net_library/services/Weight_experience_space.h"
#include "sparse_net_library/services/random_attention_brain.h"

namespace sparse_net_library_test {

using sparse_net_library::SparseNet;
using sparse_net_library::Sparse_net_builder;
using sparse_net_library::Cost_function_mse;
using sparse_net_library::COST_FUNCTION_SQUARED_ERROR;
using sparse_net_library::TRANSFER_FUNCTION_IDENTITY;
using sparse_net_library::TRANSFER_FUNCTION_SELU;
using sparse_net_library::TRANSFER_FUNCTION_RELU;
using sparse_net_library::TRANSFER_FUNCTION_SIGMOID;
using sparse_net_library::Data_aggregate;
using sparse_net_library::Weight_experience_space;
using sparse_net_library::Random_attention_brain;
using rafko_mainframe::Service_context;

/*###############################################################################################
 * Testing training of a Random Attention Brain on a simple dataset
 * */
TEST_CASE("Testing Random Attention Brain on a simple dataset","[brain]"){
  google::protobuf::Arena arena;
  Service_context service_context = Service_context().set_step_size(1e-2).set_arena_ptr(&arena);
  std::cout << "Testing a simple dataset:" << std::endl;

  /* Create a Network and Dataset */
  SparseNet* net = Sparse_net_builder(service_context)
    .input_size(2).expected_input_range(double_literal(1.0))
    .allowed_transfer_functions_by_layer(
      {
        {TRANSFER_FUNCTION_SELU}
      }
    ).dense_layers({1});

  /* Create dataset, test set and aprroximizer */
  Data_aggregate* train_set = create_addition_dataset(5, *net, COST_FUNCTION_SQUARED_ERROR, service_context);

  /* Create a Brain */
  Random_attention_brain brain(*net,*train_set,service_context);

  /* Add impulses into the bain until the error rate is sufficient */
  sdouble32 min_error = double_literal(9999.0);
  while(service_context.get_step_size() <= train_set->get_error_avg()){
    brain.step();
    std::cout << "\rError: " << train_set->get_error_avg() << "   ";
    if(min_error > train_set->get_error_avg()){
      min_error = train_set->get_error_avg();
      std::cout << "| minimum: " << min_error;
    }
  }
  std::cout << std::endl << "---" << std::endl;
}

/*###############################################################################################
 * Testing training of a Random Attention Brain on a simple dataset
 * */
TEST_CASE("Testing Random Attention Brain on a more complex, time series dataset","[brain]"){
  google::protobuf::Arena arena;
  Service_context service_context = Service_context().set_step_size(1e-2).set_arena_ptr(&arena);
  std::cout << "Testing a time-series dataset(binary addition):" << std::endl;

  /* Create a Network and Dataset */
  SparseNet* net = Sparse_net_builder(service_context)
    .input_size(2).expected_input_range(double_literal(1.0))
    .set_recurrence_to_layer()
    .allowed_transfer_functions_by_layer(
      {
        {TRANSFER_FUNCTION_SELU},
        {TRANSFER_FUNCTION_SELU}
      }
    ).dense_layers({2,1});

  /* Create dataset, test set and aprroximizer */
  Data_aggregate* train_set = create_sequenced_addition_dataset(5, 3, *net, COST_FUNCTION_SQUARED_ERROR, service_context);

  /* Create a Brain */
  Random_attention_brain brain(*net,*train_set,service_context);

  /* Add impulses into the bain until the error rate is sufficient */
  sdouble32 min_error = double_literal(9999.0);
  while(service_context.get_step_size() <= train_set->get_error_avg()){
    brain.step();
    std::cout << "\rError: " << train_set->get_error_avg() << "   ";
    if(min_error > train_set->get_error_avg()){
      min_error = train_set->get_error_avg();
      std::cout << "| minimum: " << min_error;
    }
  }
  std::cout << std::endl << "---" << std::endl;

}

} /* namespace sparse_net_library_test */
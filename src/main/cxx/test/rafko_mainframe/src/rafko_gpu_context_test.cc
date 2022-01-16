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
#include <memory>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "rafko_net/services/rafko_net_builder.h"
#include "rafko_net/services/solution_builder.h"
#include "rafko_net/services/solution_solver.h"
#include "rafko_mainframe/services/rafko_gpu_context.h"
#include "test/test_utility.h"

namespace rafko_gym_test {

TEST_CASE("Testing if GPU Context is able to build a valid openCL environment", "[context][GPU]"){
  rafko_mainframe::RafkoSettings settings;
  rafko_net::RafkoNet* network = rafko_test::generate_random_net_with_softmax_features(1u, settings);
  std::unique_ptr<rafko_mainframe::RafkoGPUContext> context;
  REQUIRE_NOTHROW(
    context = (
      rafko_mainframe::RafkoGPUContext::Builder(*network, settings)
        .select_platform().select_device()
        .build()
    )
  );
}

TEST_CASE("Testing if a standalone solution is working as intended with the GPU context","[context][GPU]"){
  google::protobuf::Arena arena;
  uint32 sequence_size = 6u;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  rafko_net::RafkoNet* network = rafko_net::RafkoNetBuilder(settings)
    .input_size(2).expected_input_range(double_literal(1.0))
    .allowed_transfer_functions_by_layer(
      {
        {rafko_net::transfer_function_relu}
      }
    ).dense_layers({1});
  std::unique_ptr<rafko_mainframe::RafkoGPUContext> context;
  REQUIRE_NOTHROW(
    context = (
      rafko_mainframe::RafkoGPUContext::Builder(*network, settings)
        .select_platform().select_device()
        .build()
    )
  );

  std::unique_ptr<rafko_net::Solution> reference_solution = rafko_net::SolutionBuilder(settings).build(*network);
  std::unique_ptr<rafko_net::SolutionSolver> reference_agent = rafko_net::SolutionSolver::Builder(*reference_solution, settings).build();
  std::vector<sdouble32> network_input(network->input_data_size(), (rand()%10));
  rafko_utilities::ConstVectorSubrange<> reference_result = reference_agent->solve(network_input);
  rafko_utilities::ConstVectorSubrange<> context_result = context->solve(network_input);

  for(uint32 result_index = 0; result_index < reference_result.size(); ++result_index){
    CHECK( reference_result[result_index] == context_result[result_index] );
  }

  (void)rafko_net::SolutionBuilder::get_kernel_for_solution(*reference_solution, "aw_yiss", settings);
}

} /* namespace rako_gym_test */

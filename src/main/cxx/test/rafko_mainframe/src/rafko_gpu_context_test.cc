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

#include "rafko_mainframe/services/rafko_gpu_context.h"
#include "test/test_utility.h"

namespace rafko_gym_test {

TEST_CASE("Testing if GPU Context is able to build a valig openCL environment", "[context][GPU]"){
  uint32 sample_number = 50;
  uint32 sequence_size = 6;
  google::protobuf::Arena arena;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  sdouble32 expected_label = double_literal(50.0);
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

} /* namespace rako_gym_test */

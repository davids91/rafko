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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <set>
#include <vector>

#include "rafko_mainframe/models/rafko_settings.hpp"
#include "rafko_net/models/transfer_function.hpp"
#include "rafko_protocol/rafko_net.pb.h"

#include "test/test_utility.hpp"

namespace rafko_net_test {

/*###############################################################################################
 * Testing Transfer function outputs
 * */
TEST_CASE("Testing Transfer function outputs", "[neuron][transfer-function]") {
  rafko_mainframe::RafkoSettings settings;
  rafko_net::TransferFunction tfun(settings);
  double data;
  for (std::uint32_t variant = 0; variant < 10u; ++variant) {
    data = static_cast<double>(rand() % 100);
    REQUIRE(tfun.get_value(rafko_net::transfer_function_identity, data) ==
            Catch::Approx(data).epsilon(0.0000000001));
    REQUIRE(tfun.get_value(rafko_net::transfer_function_sigmoid, data) ==
            Catch::Approx((1.0) / ((1.0) + exp(-data))).epsilon(0.0000000001));
    REQUIRE(tfun.get_value(rafko_net::transfer_function_elu, data) ==
            Catch::Approx(std::max((0.0), data) + std::min((0.0), data) *
                                                      settings.get_alpha() *
                                                      (std::exp(data) - 1.0))
                .epsilon(0.0000000001));
    REQUIRE(tfun.get_value(rafko_net::transfer_function_selu, data) ==
            Catch::Approx((settings.get_lambda() * std::max(0.0, data)) +
                          (std::min((0.0), data) * settings.get_lambda() *
                           settings.get_alpha() * (std::exp(data) - 1.0)))
                .epsilon(0.0000000001));
    REQUIRE(tfun.get_value(rafko_net::transfer_function_relu, data) ==
            Catch::Approx(std::max((0.0), data)).epsilon(0.0000000001));
  }
}

TEST_CASE("Testing transfer function generators",
          "[neuron][transfer-function]") {
  rafko_mainframe::RafkoSettings settings;
  rafko_net::TransferFunction tfun(settings);
  for (std::uint32_t variant = 0; variant < 10u; ++variant) {
    std::set<rafko_net::Transfer_functions> used_functions;
    while (used_functions.size() < 3) {
      std::uint32_t candidate = rand() % rafko_net::transfer_function_end;
      if (rafko_net::Transfer_functions_IsValid(candidate)) {
        used_functions.insert(
            static_cast<rafko_net::Transfer_functions>(candidate));
      }
    } /* while(a test set is filled) */

    std::vector<rafko_net::Transfer_functions> generated;
    for (std::uint32_t sequence = 0; sequence < 10; ++sequence) {
      generated.push_back(tfun.next(used_functions));
    }

    for (const rafko_net::Transfer_functions &f : generated) {
      REQUIRE(used_functions.find(f) != used_functions.end());
    }

  } /*for(10 variants)*/
}

} /* namespace rafko_net_test */

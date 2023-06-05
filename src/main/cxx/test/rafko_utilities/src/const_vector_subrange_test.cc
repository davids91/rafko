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

#include <catch2/catch_test_macros.hpp>
#include <iterator>
#include <vector>

#include "rafko_utilities/models/const_vector_subrange.hpp"

#include "test/test_utility.hpp"

namespace rafko_utilities_test {

TEST_CASE("Testing Vector subrange", "[data-handling][sub-range]") {
  std::vector<double> big_vec = {0, 1, 2,  3,  4,  5,  6,  7,
                                 8, 9, 10, 11, 12, 13, 14, 15};
  rafko_utilities::ConstVectorSubrange<> my_range{big_vec.begin(),
                                                  big_vec.end()};

  REQUIRE(big_vec.size() == my_range.size());
  REQUIRE(std::next(big_vec.end(), -1) == std::next(my_range.end(), -1));
  CHECK(big_vec.back() == my_range.back());
  CHECK(big_vec.front() == my_range.front());

  for (std::uint32_t i = 0; i < big_vec.size(); i++) {
    REQUIRE(big_vec[i] == my_range[i]);
  }

  for (std::uint32_t variant = 0; variant < 10u; variant++) {
    std::uint16_t start = rand() % big_vec.size();
    std::uint16_t num = rand() % (big_vec.size() - start) + 1;

    rafko_utilities::ConstVectorSubrange<> range =
        rafko_utilities::ConstVectorSubrange<>(big_vec.begin() + start, num);

    REQUIRE(big_vec[start] == range.front());
    REQUIRE(big_vec[start + num - 1] ==
            range.back()); /* The last is at index (start + size - 1) */

    for (int i = 0; i < num; i++) {
      REQUIRE(big_vec[start + i] == range[i]);
    }
  }
}

TEST_CASE("Testing Vector subrange Equality operator",
          "[data-handling][sub-range]") {
  std::vector<double> big_vec = {0, 1, 2,  3,  4,  5,  6,  7,
                                 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<double> big_vec_copy = big_vec;
  std::vector<double> another_vec = {big_vec.begin() + 5, big_vec.end()};
  rafko_utilities::ConstVectorSubrange<> big_range{big_vec.begin(),
                                                   big_vec.end()};
  rafko_utilities::ConstVectorSubrange<> smaller_range{big_vec.begin() + 5,
                                                       big_vec.end()};

  REQUIRE(big_range == big_vec);
  REQUIRE(big_range == big_vec_copy);
  REQUIRE(smaller_range == another_vec);
}

} /* namespace rafko_utilities_test */

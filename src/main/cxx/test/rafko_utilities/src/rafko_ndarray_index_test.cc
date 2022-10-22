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

#include <vector>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "rafko_utilities/services/rafko_math_utils.hpp"

#include "test/test_utility.hpp"

namespace rafko_utilities_test {

TEST_CASE("Testing NDArray Indexing with a 2D array without padding", "[NDArray]"){
  std::uint32_t width = rand()%100;
  std::uint32_t height = rand()%100;
  rafko_utilities::NDArrayIndex idx({width, height});

  for(std::uint32_t variant = 0; variant < 5; ++variant){
    std::uint32_t x = rand()%width;
    std::uint32_t y = rand()%height;
    idx.set({x,y});
    REQUIRE(idx.inside_bounds());
    REQUIRE(idx.mapped_position().has_value());
    REQUIRE(idx.mapped_position().value() == (x + (y * width)));
    std::uint32_t elements_after_x_row = width - x;
    REQUIRE(1 == idx.mappable_parts_of(0,width).size());
    REQUIRE(x == std::get<0>(idx.mappable_parts_of(0,width)[0]));
    REQUIRE(elements_after_x_row == std::get<1>(idx.mappable_parts_of(0,width)[0]));
    /*!Note: using width in the above interfaces because it is guaranteed
     * that an interval of that size spans over the relevant dimension
     * */
  }

  REQUIRE(idx.buffer_size() == (width * height));
  idx.set({0,0});
  for(std::uint32_t i = 0; i < idx.buffer_size(); ++i){
    REQUIRE(idx.inside_bounds());
    REQUIRE(idx.inside_content());
    REQUIRE(idx.mapped_position().has_value() == true);
    REQUIRE(idx.mapped_position().value() == i);
    idx.step();
  }
}

TEST_CASE("Testing NDArray Indexing with a 2D array with positive padding", "[NDArray][padding]"){
  std::uint32_t width = 1 + rand()%20;
  std::uint32_t height = 1 + rand()%20;
  std::int32_t padding = 5;
  rafko_utilities::NDArrayIndex idx({width, height}, padding);

  for(std::uint32_t variant = 0; variant < 5; ++variant){
    std::uint32_t x = padding + rand()%(width);
    std::uint32_t y = padding + rand()%(height);
    idx.set({x,y});
    REQUIRE(idx.inside_bounds());
    REQUIRE(idx.mapped_position().has_value());
    REQUIRE( idx.mapped_position().value() == (x - padding + ((y - padding) * width)) );
    std::uint32_t elements_after_x_row = padding + width - x;
    REQUIRE(1 == idx.mappable_parts_of(0,width).size());
    REQUIRE(x == std::get<0>(idx.mappable_parts_of(0,width)[0]));
    REQUIRE(elements_after_x_row == std::get<1>(idx.mappable_parts_of(0,width)[0]));
  }

  REQUIRE(idx.buffer_size() == (width * height));
  std::uint32_t x = 0u;
  std::uint32_t y = 0u;
  std::uint32_t reference_mapped_position = 0u;
  idx.set({0,0});
  for(std::uint32_t i = 0; i < idx.buffer_size(); ++i){
    if(
      (padding <= static_cast<std::int32_t>(x) && x < (padding + width))
      &&(padding <= static_cast<std::int32_t>(y) && y < (padding + height))      
    ){
      REQUIRE(idx.inside_bounds());
      REQUIRE(idx.inside_content());
      REQUIRE(idx.mapped_position().has_value() == true);
      REQUIRE(idx.mapped_position().value() == reference_mapped_position);
      ++reference_mapped_position;
    }else{
      REQUIRE(idx.inside_bounds());
      REQUIRE(idx.mapped_position().has_value() == false);
    } 
    idx.step();
    if(x < padding + width + padding - 1){
      ++x;
    }else{
      x = 0;
      ++y;
    }
  }
}

TEST_CASE("Testing NDArray Indexing with a 2D array with negative padding", "[NDArray][padding]"){
  std::uint32_t width = 11 + rand()%20;
  std::uint32_t height = 11 + rand()%20;
  std::int32_t padding = -5;
  rafko_utilities::NDArrayIndex idx({width, height}, padding);

  for(std::uint32_t variant = 0; variant < 5; ++variant){
    std::uint32_t x = -padding + rand()%(width + 2 * padding);
    std::uint32_t y = -padding + rand()%(height + 2 * padding);
    idx.set({x,y});

    REQUIRE(idx.inside_bounds());
    REQUIRE(idx.mapped_position().has_value());
    REQUIRE( idx.mapped_position().value() == (x + padding + ((y + padding) * (width + 2 * padding))) );
    std::uint32_t elements_after_x_row = padding + width - x;
    REQUIRE(1 == idx.mappable_parts_of(0,width).size());
    REQUIRE(x == std::get<0>(idx.mappable_parts_of(0,width)[0]));
    REQUIRE(elements_after_x_row == std::get<1>(idx.mappable_parts_of(0,width)[0]));
  }

  REQUIRE(idx.buffer_size() == (width * height));
  std::uint32_t x = 0u;
  std::uint32_t y = 0u;
  std::uint32_t reference_mapped_position = 0u;
  idx.set({0,0});
  for(std::uint32_t i = 0; i < idx.buffer_size(); ++i){
    if(
      (-padding <= static_cast<std::int32_t>(x) && x < (padding + width))
      &&(-padding <= static_cast<std::int32_t>(y) && y < (padding + height))      
    ){
      REQUIRE(idx.inside_bounds());
      REQUIRE(idx.inside_content());
      REQUIRE(idx.mapped_position().has_value() == true);
      REQUIRE(idx.mapped_position().value() == reference_mapped_position);
      ++reference_mapped_position;
    }else{
      REQUIRE(idx.inside_bounds());
      REQUIRE(idx.mapped_position().has_value() == false);
    } 
    idx.step();
    if(x < (width - 1)){
      ++x;
    }else{
      x = 0;
      ++y;
    }
  }
}

} /* namespace rafko_utilities_test */

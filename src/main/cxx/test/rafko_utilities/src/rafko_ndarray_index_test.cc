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
  REQUIRE(!idx.has_padding());
  for(std::uint32_t variant = 0; variant < 5; ++variant){
    std::uint32_t x = rand()%width;
    std::uint32_t y = rand()%height;
    idx.set({x,y});
    REQUIRE(idx.inside_bounds());
    REQUIRE(idx.mapped_position().has_value());
    REQUIRE(idx.mapped_position().value() == (x + (y * width)));
    std::uint32_t elements_after_x_row = width - x;
    REQUIRE(1 == idx.mappable_parts_of(0,width).size());
    REQUIRE(x == idx.mappable_parts_of(0,width)[0].position_start);
    REQUIRE(elements_after_x_row == idx.mappable_parts_of(0,width)[0].steps_inside_target);
    /*!Note: using width in the above interfaces because it is guaranteed
     * that an interval of that size spans over the relevant dimension
     * */
    if(y < (height - 1u))REQUIRE( idx.step(1,1).mapped_position() == (x + ((y + 1) * width)) );
      else CHECK_THROWS(idx.step(1,1));
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
  std::int32_t padding_x = rand()%5;
  std::int32_t padding_y = rand()%5;
  rafko_utilities::NDArrayIndex idx({width, height}, {padding_x, padding_y});
  REQUIRE(idx.has_padding());
  for(std::uint32_t variant = 0; variant < 5; ++variant){
    std::uint32_t x = padding_x + rand()%(width);
    std::uint32_t y = padding_y + rand()%(height);
    idx.set({x,y});
    REQUIRE(idx.inside_bounds());
    REQUIRE(idx.mapped_position().has_value());
    REQUIRE( idx.mapped_position().value() == (x - padding_x + ((y - padding_y) * width)) );
    std::uint32_t elements_after_x_row = padding_x + width - x;
    REQUIRE(1 == idx.mappable_parts_of(0,width).size());
      REQUIRE(x == idx.mappable_parts_of(0,width)[0].position_start);
    REQUIRE(elements_after_x_row == idx.mappable_parts_of(0,width)[0].steps_inside_target);
    if((static_cast<std::int32_t>(y) >= padding_y) && (y < (height + padding_y - 1)))
      REQUIRE( idx.step(1,1).mapped_position() == (x - padding_x + ((y - padding_y + 1) * width)) );
      else CHECK_NOTHROW(idx.step(1,1));
  }

  REQUIRE(idx.buffer_size() == (width * height));
  std::uint32_t x = 0u;
  std::uint32_t y = 0u;
  std::uint32_t reference_mapped_position = 0u;
  idx.set({0,0});
  for(std::uint32_t i = 0; i < idx.buffer_size(); ++i){
    if(
      (padding_x <= static_cast<std::int32_t>(x) && x < (padding_x + width))
      &&(padding_y <= static_cast<std::int32_t>(y) && y < (padding_y + height))
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
    if(x < padding_x + width + padding_x - 1){
      REQUIRE(idx.step() == 0u);
      ++x;
    }else{
      REQUIRE(idx.step() == 1u);
      x = 0;
      ++y;
    }
  }
}

TEST_CASE("Testing NDArray Indexing with a 2D array with negative padding", "[NDArray][padding]"){
  std::uint32_t width = 11 + rand()%20;
  std::uint32_t height = 11 + rand()%20;
  std::int32_t padding_x = -rand()%5;
  std::int32_t padding_y = -rand()%5;

  rafko_utilities::NDArrayIndex idx({width, height}, {padding_x, padding_y});
  REQUIRE(idx.has_padding());
  for(std::uint32_t variant = 0; variant < 5; ++variant){
    std::uint32_t x = -padding_x + rand()%(width + 2 * padding_x);
    std::uint32_t y = -padding_y + rand()%(height + 2 * padding_y);
    idx.set({x,y});

    REQUIRE(idx.inside_bounds());
    REQUIRE(idx.mapped_position().has_value());
    REQUIRE( idx.mapped_position().value() == (x + padding_x + ((y + padding_y) * (width + 2 * padding_x))) );
    std::uint32_t elements_after_x_row = padding_x + width - x;
    REQUIRE(1 == idx.mappable_parts_of(0,width).size());
    REQUIRE(x == idx.mappable_parts_of(0,width)[0].position_start);
    REQUIRE(elements_after_x_row == idx.mappable_parts_of(0,width)[0].steps_inside_target);
    if((static_cast<std::int32_t>(y) > -padding_y) && (y < (height + padding_y - 1)))
      REQUIRE( idx.step(1,1).mapped_position() == (x + padding_x + ((y + padding_y + 1) * (width + 2 * padding_x))) );
      else CHECK_NOTHROW(idx.step(1,1));
  }

  REQUIRE(idx.buffer_size() == (width * height));
  std::uint32_t x = 0u;
  std::uint32_t y = 0u;
  std::uint32_t reference_mapped_position = 0u;
  idx.set({0,0});
  for(std::uint32_t i = 0; i < idx.buffer_size(); ++i){
    if(
      (-padding_x <= static_cast<std::int32_t>(x) && x < (padding_x + width))
      &&(-padding_y <= static_cast<std::int32_t>(y) && y < (padding_y + height))
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
    if(x < (width - 1)){
      REQUIRE(idx.step() == 0u);
      ++x;
    }else{
      REQUIRE(idx.step() == 1u);
      x = 0;
      ++y;
    }
  }
}

} /* namespace rafko_utilities_test */

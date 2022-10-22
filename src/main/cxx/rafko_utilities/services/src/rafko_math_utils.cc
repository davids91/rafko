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

#include "rafko_utilities/services/rafko_math_utils.hpp"

#include <cassert>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace {

std::vector<std::uint32_t> init_strides(const std::vector<std::uint32_t>& dimensions, std::int32_t padding){
  std::vector<std::uint32_t> strides;
  std::uint32_t prev_stride = 1u;
  for(const std::uint32_t& dim : dimensions){
    strides.push_back(prev_stride);
    prev_stride *= dim + 2 * std::min(0, padding);
  }
  return strides;
}

std::vector<std::uint32_t> init_position(const std::vector<std::uint32_t>& dimensions, std::initializer_list<std::uint32_t> position){
  if(0 < position.size()){
    assert(dimensions.size() == position.size());
    return {position};
  }
  return std::vector<std::uint32_t>(dimensions.size(), 0);
}

} /* namespace */

namespace rafko_utilities {

NDArrayIndex::NDArrayIndex(
  std::initializer_list<std::uint32_t> dimensions, std::int32_t padding, 
  std::initializer_list<std::uint32_t> position
)
: m_dimensions(dimensions)
, m_padding(padding)
, m_strides(init_strides(dimensions, m_padding))
, m_bufferSize(std::accumulate(m_dimensions.begin(), m_dimensions.end(), 1.0, 
  [](const std::uint32_t& partial, const std::uint32_t& element){ return partial * element; }
))
, m_position(init_position(m_dimensions, position))
, m_mappedIndex(calculate_mapped_position(m_position))
{
  assert(0 == std::count(m_dimensions.begin(), m_dimensions.end(), 0));
  assert(inside_bounds(m_position));
}

NDArrayIndex& NDArrayIndex::set(const std::vector<std::uint32_t>& position){
  assert(position.size() == m_position.size());
  assert(inside_bounds(position));
  m_position = position;
  m_mappedIndex = calculate_mapped_position(m_position);
  assert( (!m_mappedIndex.has_value())||(m_mappedIndex.value() < m_bufferSize) );
  return *this;
}

NDArrayIndex& NDArrayIndex::step(){
  std::uint32_t dim = 0;
  bool changed = false;
  while(dim < m_dimensions.size()){
    if(inside_bounds(dim, 1)){
      step(dim, 1);
      break;
    }else{
      changed = true;
      m_position[dim] = 0;
    }
    ++dim;
  }
  if(dim >= m_dimensions.size()){
    m_mappedIndex = 0; /* Overflow happened, start from the beginning */
  }else{
    if(changed)m_mappedIndex = calculate_mapped_position(m_position);
    assert(m_mappedIndex < m_bufferSize);
  }
  return *this;
}

NDArrayIndex& NDArrayIndex::step(std::uint32_t dimension, std::int32_t delta){
  const std::int32_t new_position = static_cast<std::int32_t>(m_position[dimension]) + delta;
  assert(0 <= new_position);
  assert((m_dimensions[dimension] + (2 * std::max(0, m_padding))) > static_cast<std::uint32_t>(new_position));

  m_position[dimension] = new_position;

  bool new_position_is_inside_content = inside_content(m_position);
  if(m_mappedIndex.has_value() && new_position_is_inside_content){ /* m_mappedIndex has a value if the previous position was valid */
    m_mappedIndex.value() += m_strides[dimension] * delta;
    assert(m_mappedIndex < m_bufferSize);
  }else if(new_position_is_inside_content){ /* if the new position is inside bounds, then the mapped index can be caluclated */
    m_mappedIndex = calculate_mapped_position(m_position);
  }else m_mappedIndex = {}; /* No mapped index for positions inside the padding */
  return *this;
}

std::optional<std::uint32_t> NDArrayIndex::calculate_mapped_position(const std::vector<std::uint32_t>& position) const{
  assert(position.size() == m_strides.size());
  if(!inside_content(position))
    return {};

  std::uint32_t result_index = 0u;
  for(std::uint32_t dim = 0; dim < position.size(); ++dim){
    result_index += (position[dim] - std::max(m_padding, -m_padding)) * m_strides[dim];
  }
  return result_index;
}

bool NDArrayIndex::inside_bounds(const std::vector<std::uint32_t>& position, std::uint32_t dimension, std::int32_t delta) const{
  std::uint32_t dimension_index = 0;
  return std::all_of(position.begin(), position.end(), 
    [this, &dimension_index, dimension, delta](const std::uint32_t& pos){
      std::int32_t position = static_cast<std::int32_t>(pos);
      if(dimension_index == dimension) position += delta;
      return( (0 <= position)&&(position < static_cast<int32_t>(2 * std::max(0, m_padding) + m_dimensions[dimension_index++])) );
    }
  );
}

bool NDArrayIndex::inside_content(const std::vector<std::uint32_t>& position, std::uint32_t dimension, std::int32_t delta) const{
  std::uint32_t dimension_index = 0;
  return std::all_of(position.begin(), position.end(), 
    [this, &dimension_index, dimension, delta](const std::uint32_t& pos){
      std::int32_t actual_position = static_cast<std::int32_t>(pos);
      if(dimension_index == dimension) actual_position += delta;
      return( 
        (std::max(m_padding, -m_padding) <= actual_position)
        &&(actual_position < static_cast<std::int32_t>(m_dimensions[dimension_index++] + m_padding)) 
      );
    }
  );
}

std::vector<NDArrayIndex::IntervalPart> NDArrayIndex::mappable_parts_of(
  const std::vector<std::uint32_t>& position, std::uint32_t dimension, std::int32_t delta
) const{
  std::vector<NDArrayIndex::IntervalPart> result;
  bool part_in_progress = false;
  for(std::int32_t delta_index = 0; delta_index < delta; delta_index += std::copysign(1, delta)){
    const bool current_position_in_inside_content = inside_content(position, dimension, delta_index);
    if(current_position_in_inside_content && part_in_progress){
      assert(0 < result.size());
      ++std::get<1>(result.back()); /* Increase the size of the current part of the interval */
    }else if(current_position_in_inside_content){ /* If the interval iteration became inside bounds */
      result.push_back({(position[dimension] + delta_index), 1}); /* Add the new part as a result */
      part_in_progress = true;
    }else part_in_progress = false;
  }
  return result;
}


} /* namespace rafko_utilities */

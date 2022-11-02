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

std::vector<std::int32_t> init_padding(
  const std::vector<std::uint32_t>& dimensions, const std::vector<std::int32_t>& padding
){
  if(1 == padding.size())
    return std::vector<std::int32_t>(dimensions.size(), padding[0]);

  if(1 < padding.size()){
    assert(dimensions.size() == padding.size());
    return padding;
  }
  return std::vector<std::int32_t>(dimensions.size(), 0);
}

std::vector<std::uint32_t> init_strides(
  const std::vector<std::uint32_t>& dimensions, const std::vector<std::int32_t>& padding
){
  assert(dimensions.size() == padding.size());
  if(0 == dimensions.size()) return {};
  std::vector<std::uint32_t> strides;
  std::uint32_t prev_stride = 1u;
  std::int32_t prev_padding = padding[0];
  std::uint32_t dim = 0;
  for(const std::uint32_t& dimension : dimensions){
    strides.push_back(prev_stride);
    prev_stride *= dimension + 2 * std::min(0, prev_padding);
    prev_padding = padding[dim++];
  }
  return strides;
}

std::vector<std::uint32_t> init_position(
  const std::vector<std::uint32_t>& dimensions, const std::vector<std::uint32_t>& position
){
  if(0 < position.size()){
    assert(dimensions.size() == position.size());
    return {position};
  }
  return std::vector<std::uint32_t>(dimensions.size(), 0);
}

} /* namespace */

namespace rafko_utilities {

NDArrayIndex::NDArrayIndex(
  const std::vector<std::uint32_t>& dimensions, const std::vector<std::int32_t>& padding,
  const std::vector<std::uint32_t>& position
)
: m_dimensions(dimensions)
, m_padding(init_padding(m_dimensions, padding))
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

NDArrayIndex::NDArrayIndex(const NDArrayIndex& other, const std::vector<std::int32_t>& padding)
: m_dimensions(other.m_dimensions)
, m_padding(init_padding(m_dimensions, padding))
, m_strides(init_strides(m_dimensions, m_padding))
, m_bufferSize(other.m_bufferSize)
, m_position(other.m_position)
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

NDArrayIndex& NDArrayIndex::set(std::uint32_t dimension, std::uint32_t position){
  assert(dimension < size());
  m_position[dimension] = position;
  m_mappedIndex = calculate_mapped_position(m_position);
  assert( (!m_mappedIndex.has_value())||(m_mappedIndex.value() < m_bufferSize) );
  assert(inside_bounds(m_position));
  return *this;
}


std::uint32_t NDArrayIndex::step(){
  std::uint32_t dim = 0;
  bool changed = false;
  while(dim < m_dimensions.size()){
    if(inside_bounds(dim, 1)){
      step(dim, 1);
      break;
    }else{
      changed = true;
      m_position[dim] = 0u;
    }
    ++dim;
  }

  if(dim >= m_dimensions.size()){
    m_mappedIndex = 0u; /* Overflow happened, start from the beginning */
    return m_dimensions.size() - 1;
  }else{
    if(changed)m_mappedIndex = calculate_mapped_position(m_position);
    assert(m_mappedIndex < m_bufferSize);
    return dim;
  }
}

bool NDArrayIndex::step(std::uint32_t dimension, std::int32_t delta){
  if(!inside_bounds(m_position, dimension, delta))return false;
  m_position[dimension] += delta;
  bool new_position_is_inside_content = inside_content(m_position);
  if(m_mappedIndex.has_value() && new_position_is_inside_content){ /* m_mappedIndex has a value if the previous position was valid */
    m_mappedIndex.value() += m_strides[dimension] * delta;
    assert(m_mappedIndex < m_bufferSize);
  }else if(new_position_is_inside_content){ /* if the new position is inside bounds, then the mapped index can be caluclated */
    m_mappedIndex = calculate_mapped_position(m_position);
  }else m_mappedIndex = {}; /* No mapped index for positions inside the padding */
  return true;
}

std::optional<std::uint32_t> NDArrayIndex::calculate_mapped_position(
  const std::vector<std::uint32_t>& position, std::uint32_t dimension, std::int32_t delta
) const{
  assert(position.size() == m_strides.size());
  assert(inside_bounds(m_position, dimension, delta));
  if(!inside_content(position, dimension, delta))
    return {};

  std::uint32_t result_index = 0u;
  for(std::uint32_t dim = 0; dim < position.size(); ++dim){
    std::uint32_t new_position = position[dim];
    if(dim == dimension)new_position += delta;
    result_index += (new_position - std::max(m_padding[dim], -m_padding[dim])) * m_strides[dim];
  }
  return result_index;
}

bool NDArrayIndex::inside_bounds(const std::vector<std::uint32_t>& position, std::uint32_t dimension, std::int32_t delta) const{
  std::uint32_t dim = 0;
  return std::all_of(position.begin(), position.end(), 
    [this, &dim, dimension, delta](const std::uint32_t& pos){
      std::int32_t position = static_cast<std::int32_t>(pos);
      if(dim == dimension) position += delta;
      ++dim;
      return(
        (0 <= position)
        &&( position < (static_cast<std::int64_t>(m_dimensions[dim - 1]) + 2 * std::max(std::int64_t{0}, static_cast<int64_t>(m_padding[dim - 1]))) )
      );
    }
  );
}

bool NDArrayIndex::inside_content(const std::vector<std::uint32_t>& position, std::uint32_t dimension, std::int32_t delta) const{
  std::uint32_t dim = 0;
  return std::all_of(position.begin(), position.end(), 
    [this, &dim, dimension, delta](const std::uint32_t& pos){
      std::int32_t actual_position = static_cast<std::int32_t>(pos);
      if(dim == dimension) actual_position += delta;
      ++dim;
      return(
        (std::max(m_padding[dim - 1], -m_padding[dim - 1]) <= actual_position)
        &&( actual_position < static_cast<std::int64_t>(m_dimensions[dim - 1] + m_padding[dim - 1]) )
      );
    }
  );
}

std::vector<NDArrayIndex::IntervalPart> NDArrayIndex::mappable_parts_of(
  const std::vector<std::uint32_t>& position, std::uint32_t dimension, std::int32_t delta
) const{
  std::vector<NDArrayIndex::IntervalPart> result;
  bool part_in_progress = false;
  for(std::int32_t delta_index = 0; delta_index != delta; delta_index += std::copysign(1, delta)){
    const bool current_position_in_inside_content = inside_content(position, dimension, delta_index);
    if(current_position_in_inside_content && part_in_progress){
      assert(0 < result.size());
      ++result.back().steps_inside_target; /* Increase the size of the current part of the interval */
    }else if(current_position_in_inside_content){ /* If the interval iteration became inside bounds */
      result.push_back({(position[dimension] + delta_index), 1}); /* Add the new part as a result */
      part_in_progress = true;
    }else part_in_progress = false;
  }
  return result;
}

void NDArrayIndex::scan_kernel(NDArrayIndex& kernel, std::function<void(std::uint32_t, std::uint32_t)> fun){
  assert(!kernel.has_padding());
  assert(size() == kernel.size());
  std::vector<std::uint32_t> original_position = m_position;
  kernel.reset();
  do{ /* Acquire interval inside bounds of the mappable buffer and call the function with it */
    std::vector<NDArrayIndex::IntervalPart> parts_inside_content = mappable_parts_of(m_position, 0, kernel[0]);
    if(0 < parts_inside_content.size()){
      auto& [start_pos, interval_size] = parts_inside_content[0];
      if(mapped_position().has_value()){
        fun(mapped_position().value() - m_position[0] + start_pos, interval_size);
      }else{
        step(0, start_pos);
        fun(calculate_mapped_position(m_position).value(), interval_size);
        step(0, -start_pos);
      }
    }

    if(kernel.step(1,1)){ /* Step in dimension[1], because dimension[0] should stay at position 0 */
      [[maybe_unused]]bool stepped = step(1,1); /* ...and if succesful, step with the current object as well */
      assert(stepped);
    }else{ /* In case kernel couldn't step in dimension[1] step to the next position by the wrap around interface */
      kernel.set(0, kernel[0] - 1); /* set the first dimensions position inside the kernel to the last */
      std::uint32_t modified_dimension = kernel.step();
      assert(modified_dimension < m_dimensions.size());
      assert(kernel[modified_dimension] <= (*this)[modified_dimension]);
      /*!Note: Because of overflow, the position in dimension[0] will reset to zero */
      if((kernel.mapped_position().has_value())&&(kernel.mapped_position().value() != 0)){ /* If iteration will continue */
        for(std::uint32_t dim = 0; dim < modified_dimension; ++dim){ /* set the dimension position accordingly */
          m_position[dim] = original_position[dim];
        }
        ++m_position[modified_dimension];
        m_mappedIndex = calculate_mapped_position(m_position);
      }
    }
  }while( /* when the mapped position for the kernel points to the start of the first dimension, and the end of the others */
    (kernel.mapped_position().has_value()) /* the kernel is iterated through */
    &&(0 != kernel.mapped_position().value())
  );
  set(original_position);
}

} /* namespace rafko_utilities */

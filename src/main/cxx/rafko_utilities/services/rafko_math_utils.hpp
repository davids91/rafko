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

#ifndef RAFKO_MATH_UTILS_H
#define RAFKO_MATH_UTILS_H

#include "rafko_global.hpp"

#include <cstdint>
#include <vector>
#include <optional>

namespace rafko_utilities{

/**
 * @brief   Provides a unique hash for a provided pair based on value
 *
 * @param[in]   input   the pair to generate the hash for
 *
 * @return    the generated hash
 */
static constexpr std::uint64_t pair_hash(std::pair<std::int32_t,std::uint32_t> input){
  return(static_cast<std::uint64_t>(std::get<0>(input))|(static_cast<std::uint64_t>(std::get<1>(input)) << 32u));
}

class NDArrayIndex{
public:
  NDArrayIndex(
    std::initializer_list<std::uint32_t> dimensions, std::int32_t padding = 0, 
    std::initializer_list<std::uint32_t> position = {}
  );

  NDArrayIndex& set(const std::vector<std::uint32_t>& position);
  NDArrayIndex& step();
  NDArrayIndex& step(std::uint32_t dimension, std::int32_t delta);
  const std::vector<std::uint32_t>& position() const{
    return m_position;
  }
  std::optional<std::uint32_t> calculate_mapped_position(const std::vector<std::uint32_t>& position) const;
  std::optional<std::uint32_t> mapped_position() const{
    return m_mappedIndex;
  }
  bool inside_bounds(const std::vector<std::uint32_t>& position, std::uint32_t dimension = 0u, std::int32_t delta = 0) const;
  bool inside_bounds(std::uint32_t dimension = 0u, std::int32_t delta = 0) const{
    return inside_bounds(m_position, dimension, delta);
  }
  bool inside_bounds(const NDArrayIndex& index, std::uint32_t dimension = 0u, std::int32_t delta = 0) const{
    return inside_bounds(index.position(), dimension, delta);
  }
  bool inside_content(const std::vector<std::uint32_t>& position, std::uint32_t dimension = 0u, std::int32_t delta = 0) const;
  bool inside_content(std::uint32_t dimension = 0u, std::int32_t delta = 0) const{
    return inside_content(m_position, dimension, delta);
  }
  bool inside_content(const NDArrayIndex& index, std::uint32_t dimension = 0u, std::int32_t delta = 0) const{
    return inside_content(index.position(), dimension, delta);
  }

  using IntervalPart = std::pair<std::uint32_t, std::uint32_t>;
  std::vector<IntervalPart> mappable_parts_of(std::uint32_t dimension, std::int32_t delta) const{
    return mappable_parts_of(m_position, dimension, delta);
  }
  std::vector<IntervalPart> mappable_parts_of(
    const std::vector<std::uint32_t>& position, std::uint32_t dimension, std::int32_t delta
  ) const;

  std::uint32_t buffer_size(){
    return m_bufferSize;
  }

private:
  const std::vector<std::uint32_t> m_dimensions;
  const std::int32_t m_padding;
  const std::vector<std::uint32_t> m_strides;
  const std::uint32_t m_bufferSize;
  std::vector<std::uint32_t> m_position;
  std::optional<std::uint32_t> m_mappedIndex;
};

} /* namespace rafko_utilities */

#endif/* RAFKO_MATH_UTILS_H */

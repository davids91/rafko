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

#ifndef CONST_VECTOR_SUBRANGE_H
#define CONST_VECTOR_SUBRANGE_H

#include "rafko_global.hpp"

#include <vector>
#include <cassert>

namespace rafko_utilities{

/** 
 * @brief     A lightweight class representing a read-only part of a vector-like container,  
 *            using iterators to simulate the interface of it.
 */
template <typename Iterator = std::vector<double>::const_iterator>
class RAFKO_EXPORT ConstVectorSubrange{
public:
  using T = typename Iterator::value_type;

  ConstVectorSubrange(const std::vector<T>& data)
  : m_start(data.begin())
  , m_rangeSize(data.size())
  {
  }

  constexpr ConstVectorSubrange(Iterator start, std::size_t size)
  : m_start(start)
  , m_rangeSize(size)
  {
  }

  constexpr ConstVectorSubrange(Iterator begin, Iterator end)
  : m_start(begin)
  , m_rangeSize(std::distance(m_start, end))
  {
  }

  const T& operator[](std::size_t index) const{
    assert(index < m_rangeSize);
    return *std::next(m_start, index);
  }
  constexpr const T& front() const{
    return *begin();
  }
  constexpr const T& back() const{
    return *std::next(end(), -1);
  }
  constexpr std::size_t size() const{
    return m_rangeSize;
  }
  constexpr Iterator begin() const{
    return m_start;
  }
  constexpr Iterator end() const{
    return std::next(m_start, m_rangeSize);
  }

  template <typename V = std::vector<T>>
  constexpr V acquire(){
    return { begin(), end() };
  }

private:
  const std::vector<T> m_maybeData;
  const Iterator m_start;
  const std::size_t m_rangeSize;
};

} /* namespace rafko_utilities */

#endif/* CONST_VECTOR_SUBRANGE_H */

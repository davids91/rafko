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

#include "rafko_global.h"

#include <vector>
#include <cassert>

namespace rafko_utilities{

template <typename Iterator = std::vector<double>::const_iterator>
class RAFKO_FULL_EXPORT ConstVectorSubrange{
public:
  using T = typename Iterator::value_type;

  constexpr ConstVectorSubrange(Iterator start_, std::size_t size_)
  : start(start_)
  , range_size(size_)
  { }

  constexpr ConstVectorSubrange(Iterator begin_, Iterator end_)
  : start(begin_)
  , range_size(std::distance(start, end_))
  { }

  const T& operator[](std::size_t index) const{
    assert(index < range_size);
    return *std::next(start, index);
  }
  constexpr const T& front() const{
    return *begin();
  }
  constexpr const T& back() const{
    return *std::next(end(), -1);
  }
  constexpr std::size_t size() const{
    return range_size;
  }
  constexpr Iterator begin() const{
    return start;
  }
  constexpr Iterator end() const{
    return std::next(start, range_size);
  }

  constexpr std::vector<typename std::iterator_traits<Iterator>::value_type> as_vector(){
    return {begin(),end()};
  }

private:
  const Iterator start;
  const std::size_t range_size;
};

} /* namespace rafko_utilities */

#endif/* CONST_VECTOR_SUBRANGE_H */

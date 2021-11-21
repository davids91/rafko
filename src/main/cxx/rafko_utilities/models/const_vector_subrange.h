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

#include <iterator>
#include <vector>

namespace rafko_utilities{

template <typename T = sdouble32>
class ConstVectorSubrange{
public:
  ConstVectorSubrange(typename std::vector<T>::const_iterator start_, std::size_t size_)
  : start(start_)
  , range_size(size_)
  { }

  ConstVectorSubrange(typename std::vector<T>::const_iterator begin_, typename std::vector<T>::const_iterator end_)
  : start(begin_)
  , range_size(std::distance(start, end_))
  { }

  const T& operator[](std::size_t index) const{
    assert(index < range_size);
    return *std::next(start, index);
  }
  const T& front(void) const{
    return *cbegin();
  }
  const T& back(void) const{
    return *std::next(cend(), -1);
  }
  std::size_t size(void) const{
    return range_size;
  }
  const typename std::vector<T>::const_iterator cbegin(void) const{
    return start;
  }
  const typename std::vector<T>::const_iterator cend(void) const{
    return std::next(start, range_size);
  }

private:
  const typename std::vector<T>::const_iterator start;
  const std::size_t range_size;
};

} /* namespace rafko_utilities */

#endif/* CONST_VECTOR_SUBRANGE_H */

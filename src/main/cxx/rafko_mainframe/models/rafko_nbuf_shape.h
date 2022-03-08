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

#ifndef RAFKO_NBUF_SHAPE
#define RAFKO_NBUF_SHAPE

#include "rafko_global.h"

#include <memory>
#include <vector>
#include <CL/opencl.hpp>

namespace rafko_mainframe{

/**
 * @brief      A container class to store a shape for multiple consequitve
 *             buffers mapped onto one. The buffers are mapped into memory
 *             as if they were concatenated. Each buffer handles their own internal
 *             structure.
 */
class RAFKO_FULL_EXPORT RafkoNBufShape : public std::vector<std::size_t>{
public:
  RafkoNBufShape(std::initializer_list<std::size_t> list){
    for(const std::size_t& s : list)push_back(s);
  }

  /**
   * @brief      Provides the bytesize of a buffer of this shape with the given type T
   *
   * @return     the bytesize of a buffer of this shape with the given type T
   */
  template<typename T>
  std::size_t get_byte_size() const{
    std::size_t byte_size = 0u;
    for(const std::size_t& dim : *this)
      byte_size += (sizeof(T) * dim);
    return byte_size;
  }

  /**
   * @brief      Provides the number of elements in a buffer of this shape
   *
   * @return     the number of elements stored in a buffer of this shape
   */
  std::size_t get_number_of_elements() const{
    std::size_t number = 0u;
    for(const std::size_t& dim : *this) number += dim;
    return number;
  }

  /**
   * @brief      Provides the bytesize required to store the shape
   *
   * @return     the number of bytes required to store this shape
   */
  std::size_t get_shape_buffer_byte_size() const{
    return sizeof(cl_int) * size();
  }

  /**
   * @brief      Provides the shape of the Nbuffer in `cl_int` datatype.
   *             The bytesize of std::size_t and cl_int might differ! 
   *
   * @return     Allocated bytes representing the shape of the RafkoNDArray object
   */
  std::unique_ptr<cl_int[]> acquire_shape_buffer() const{
    std::unique_ptr<cl_int[]> shape_buffer(new cl_int[size()]);
    std::copy(begin(), end(), shape_buffer.get());
    return shape_buffer;
  }
};

} /* namespace rafko_mainframe */

#endif /* RAFKO_NBUF_SHAPE */

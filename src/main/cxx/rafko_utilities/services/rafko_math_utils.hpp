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
#include <functional>

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
    const std::vector<std::uint32_t>& dimensions, const std::vector<std::int32_t>& padding = {},
    const std::vector<std::uint32_t>& position = {}
  );
  NDArrayIndex(const NDArrayIndex& other, const std::vector<std::int32_t>& padding);

  /** @brief    Sets the position to all zeroes
   *
   * @return    Reference to the object
   */
  NDArrayIndex& reset(){
    return set(std::vector<std::uint32_t>(size(), 0));
  }

  /** @brief    updates the position of the object based on the given argument
   * 
   * @param[in]   position    The position to move the object to; Must be within bounds!
   * 
   * @return    Reference to the object
   */
  NDArrayIndex& set(const std::vector<std::uint32_t>& position);

  /** @brief    updates the position of the object based on the given arguments
   *
   * @param[in]   dimension     The dimension to set the position of; Must be within bounds!
   * @param[in]   position      The position to move the object to; Must be within bounds!
   *
   * @return    Reference to the object
   */
  NDArrayIndex& set(std::uint32_t dimension, std::uint32_t position);

  /** @brief    updates the position of the object to go to the next position in the buffer range
   * 
   * @return    The index of the highest dimension the step modified
   */
  std::uint32_t step();

  /** @brief    updates the position of the object to go to direction given in the arguments if the new position is inside bounds,
   *            throws an exception of the new position is not inside bounds
   * 
   * @param[in]   dimension    The dimension to move the object on
   * @param[in]   delta        The number of steps to move the objects position in the given dimension
   * 
   * @return    True if the step was succesful
   */
  bool step(std::uint32_t dimension, std::int32_t delta = 1);

  /** @brief    References the stored position
   *
   *  @return   a constant reference to the stored position, each element in the vector is the position in the appropriate dimension
   */
  const std::vector<std::uint32_t>& position() const{
    return m_position;
  }

  /** @brief    updates the position of the object based on the given argument
   * 
   * @param[in]   position    The position to base the calculations on
   * 
   * @return    The value of the mapped index, if there is any
   */
  std::optional<std::uint32_t> calculate_mapped_position(
    const std::vector<std::uint32_t>& position, std::uint32_t dimension = 0, std::int32_t delta = 0
  ) const;

  /** @brief    Provides the index of the current position in the underlying buffer, if there is any
   * 
   * @return    the index pointing to the actual position, if it is mappable to the internal buffer
   */  
  std::optional<std::uint32_t> mapped_position() const{
    return m_mappedIndex;
  }

  /** @brief    Tells if the given position is inside the indexing interval provided by the dimensions and padding of the object
   * 
   * @param[in]   position     The position to base the calculations on
   * @param[in]   dimension    An optional dimension parameter to help shift the parameter
   * @param[in]   delta        The number of steps to move the objects position temporarily in the given dimension for the relevant check
   * 
   * @return    true if the given position is inside bounds
   */  
  bool inside_bounds(const std::vector<std::uint32_t>& position, std::uint32_t dimension = 0u, std::int32_t delta = 0) const;
  
  /** @brief    Tells if the stored position is inside the indexing interval provided by the dimensions and padding of the object
   * 
   * @param[in]   dimension    An optional dimension parameter to help shift the parameter
   * @param[in]   delta        The number of steps to move the objects position temporarily in the given dimension for the relevant check
   * 
   * @return    true if the given position is inside bounds
   */  
  bool inside_bounds(std::uint32_t dimension = 0u, std::int32_t delta = 0) const{
    return inside_bounds(m_position, dimension, delta);
  }
  
  /** @brief    Tells if the given position is inside the indexing interval provided by the dimensions and padding of the object
   * 
   * @param[in]   index        The index object supporting the position to base the check upon
   * @param[in]   dimension    An optional dimension parameter to help shift the parameter
   * @param[in]   delta        The number of steps to move the objects position temporarily in the given dimension for the relevant check
   * 
   * @return    true if the given position is inside bounds
   */
  bool inside_bounds(const NDArrayIndex& index, std::uint32_t dimension = 0u, std::int32_t delta = 0) const{
    return inside_bounds(index.position(), dimension, delta);
  }

  /** @brief    Tells if the given position is mappable to the buffer range determined by the dimensions and padding
   * 
   * @param[in]   position     The position to base the calculations on
   * @param[in]   dimension    An optional dimension parameter to help shift the parameter
   * @param[in]   delta        The number of steps to move the objects position temporarily in the given dimension for the relevant check
   * 
   * @return    true if the given position is inside bounds
   */
  bool inside_content(const std::vector<std::uint32_t>& position, std::uint32_t dimension = 0u, std::int32_t delta = 0) const;

  /** @brief    Tells if the stored position is mappable to the buffer range determined by the dimensions and padding
   * 
   * @param[in]   dimension    An optional dimension parameter to help shift the parameter
   * @param[in]   delta        The number of steps to move the objects position temporarily in the given dimension for the relevant check
   * 
   * @return    true if the given position is inside bounds
   */
  bool inside_content(std::uint32_t dimension = 0u, std::int32_t delta = 0) const{
    return inside_content(m_position, dimension, delta);
  }

  /** @brief    Tells if the given position is mappable to the buffer range determined by the dimensions and padding
   * 
   * @param[in]   index        The index object supporting the position to base the check upon
   * @param[in]   dimension    An optional dimension parameter to help shift the parameter
   * @param[in]   delta        The number of steps to move the objects position temporarily in the given dimension for the relevant check
   * 
   * @return    true if the given position is inside bounds
   */
  bool inside_content(const NDArrayIndex& index, std::uint32_t dimension = 0u, std::int32_t delta = 0) const{
    return inside_content(index.position(), dimension, delta);
  }

  /** @struct IntervalPart
   *  @brief Describes part of an interval excluding the direction it lies on
   *  @var    position_start          the absolute starting position of the interval relevan part
   *  @var    m_stepsInsideTarget     the size of the interval's relevant part
   */
  struct IntervalPart{
    const std::uint32_t position_start;
    const std::uint32_t m_stepsInsideTarget;
  };

  /** @brief    Tells which parts of the provided range relative to the stored direction are mappable to 
   *            the buffer range determined by the dimensions and padding
   * 
   * @param[in]   position      The starting position of the provided range
   * @param[in]   dimension     The direction of the provided range
   * @param[in]   delta         The size of the range starting from the stored position
   * 
   * @return    A vector of the parts of the interval inside the bounds of the objects buffer range:
   *            {position, size}:
   *            |->absolute position inside the given dimension, 
   *            |--------->number of steps still inside the defined ranges in the direction of the given dimension
   */
  std::vector<IntervalPart> mappable_parts_of(
    const std::vector<std::uint32_t>& position, std::uint32_t dimension, std::int32_t delta
  ) const;

  /** @brief    Tells which parts of the stored range relative to the stored direction are mappable to 
   *            the buffer range determined by the dimensions and padding
   * 
   * @param[in]   dimension    The direction of the range relative to the currently stored position
   * @param[in]   delta        The size of the range starting from the stored position
   * 
   * @return    A vector of the parts of the interval inside the bounds of the objects buffer range:
   *            {position, size}:
   *            |->absolute position inside the given dimension, 
   *            |--------->number of steps still inside the defined ranges in the direction of the given dimension
   */
  std::vector<IntervalPart> mappable_parts_of(std::uint32_t dimension, std::int32_t delta) const{
    return mappable_parts_of(m_position, dimension, delta);
  }

  /** @brief    Tells the size of the internal buffer range, which maps every item in the NDArray to a one dimensional array
   * 
   * @return    Reference to the object
   */
  std::uint32_t buffer_size() const{
    return m_bufferSize;
  }

  /** @brief    Returns the number of dimensions
   *
   * @return    The number of dimensions
   */
  std::uint32_t size() const{
    return m_dimensions.size();
  }

  /** @brief    Returns the number of elements inside bounds under the given dimension
   *
   * @param[in]     dimension     the dimension to query the size for
   *
   * @return    the number of elements ( including padding ) the given dimension contains
   */
  std::uint32_t operator[](std::int32_t dimension) const{
    return m_padding[dimension] + m_dimensions[dimension] + m_padding[dimension];
  }

  /** @brief    Tells if the Object contains any padding at all
   *
   * @return    true, if any padding dimension is non-zero
   */
  bool has_padding(){
    return (static_cast<std::uint32_t>(std::count(m_padding.begin(), m_padding.end(), 0)) < m_padding.size());
  }

  /** @brief    Runs a given function through a kernel, starting from the stored position.
   *            The provided function is being called once every time kernel iteration hits the
   *            beginning of dimension[0] in the provided kernel. The arguments are called with
   *            the mapped index values inside this object, along with the count of elements available
   *            from the start until the end of dimension[0]. The position of the object is updated along
   *            with the position of the kernel during iteration, and is restored after iteration is finished.
   *
   * @param         kernel    the kernel dimensions to use for iteration
   * @param[in]     fun       the function to call for each kernel iteration.
   *                          Arguments: void(mapped_position, interval size)
   */
  void scan_kernel(NDArrayIndex& kernel, std::function<void(std::uint32_t, std::uint32_t)> fun);

private:
  const std::vector<std::uint32_t> m_dimensions;
  const std::vector<std::int32_t> m_padding;
  const std::vector<std::uint32_t> m_strides;
  const std::uint32_t m_bufferSize;
  std::vector<std::uint32_t> m_position;
  std::optional<std::uint32_t> m_mappedIndex;
};

} /* namespace rafko_utilities */

#endif/* RAFKO_MATH_UTILS_H */

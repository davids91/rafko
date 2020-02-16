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
 *    along with Foobar.  If not, see <https://www.gnu.org/licenses/> or
 *    <https://github.com/davids91/rafko/blob/master/LICENSE>
 */

#ifndef Dense_net_weight_initializer_H
#define Dense_net_weight_initializer_H

#include "models/weight_initializer.h"

namespace sparse_net_library {

/**
 * @brief      Class for dense net weight initializer. The member functions of the class are documented
 *             under @Weight_initializer. The Aim of this specialization is to provide correct weight,
 *             bias and memory filter initalization to Fully Connected(Dense) Feedforward Neural networks.
 */
class Dense_net_weight_initializer : public Weight_initializer{
public:

  /**
   * @brief      Constructs the object and calls the srand function with the given arguments.
   *             To srand with time(nullptr), the constructor needs to be called with
   *             a true boolean argument or given a seed value.
   */
  Dense_net_weight_initializer(bool seed);
  Dense_net_weight_initializer(sdouble32 memRatioMin = 0.0, sdouble32 memRatioMax = 0.0);
  Dense_net_weight_initializer(uint32 seed, sdouble32 memRatioMin = 0.0, sdouble32 memRatioMax = 0.0);

  /**
   * @brief      Configuration functions
   */
  void set(uint32 expected_input_number, sdouble32 expected_input_maximum_value_);
  sdouble32 next_weight_for(transfer_functions used_transfer_function) const;
  sdouble32 next_memory_filter() const;
  sdouble32 next_bias() const;

private:
  sdouble32 memMin = 0.0;
  sdouble32 memMax = 0.0;

  /**
   * @brief      Gets the expected amplitude for a weight with the given transfer function
   *             and configured parameters. Should always be a positive number above
   *             @Transfer_function::epsilon
   *
   * @param[in]  used_transfer_function  The basis transfer function
   *
   * @return     Expected weight amplitude
   */
  sdouble32 get_weight_amplitude(transfer_functions used_transfer_function) const;

};

} /* namespace sparse_net_library */
#endif // Dense_net_weight_initializer_H

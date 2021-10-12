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

#ifndef SPIKE_FUNCTION_H
#define SPIKE_FUNCTION_H

#include "rafko_global.h"

#include <vector>

namespace sparse_net_library{

/**
 * @brief      Spike function handling and utilities
 */
class Spike_function{
public:
  /**
   * @brief      Apply the given spike function to a neurons activation data
   *
   * @param[in]  parameter The parameter supplied by a Neuron
   * @param[in]  data      The data to apply it to
   */
  static sdouble32 get_value(sdouble32 parameter, sdouble32 new_data, sdouble32 previous_data){
    return (previous_data * parameter) + (new_data * (double_literal(1.0)-parameter));
  }

  /**
   * @brief      Gets a functions derivative calculated form the given data
   *
   * @param[in]  parameter The parameter supplied by a Neuron
   * @param[in]  data      The data to use
   *
   * @return     The derivative from data.
   */
  static sdouble32 get_derivative(sdouble32 parameter, sdouble32 transfer_function_output, sdouble32 previous_value){
    return (previous_value - transfer_function_output);
  }
};
} /* namespace sparse_net_library */
#endif /* SPIKE_FUNCTION_H */

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

#ifndef TRANSFER_FUNCTION_H
#define TRANSFER_FUNCTION_H

#include "rafko_global.h"

#include <vector>
#if(RAFKO_USES_OPENCL)
#include <string>
#endif/*(RAFKO_USES_OPENCL)*/


#include "rafko_mainframe/models/rafko_settings.h"

namespace rafko_net{

/**
 * @brief      Transfer function handling and utilities
 */
class TransferFunction{
public:
  TransferFunction(rafko_mainframe::RafkoSettings& settings)
  : settings(settings)
  { }

  /**
   * @brief      Gives a random Transfer Function
   *
   * @return     A random Transfer Function
   */
  static Transfer_functions next();

  /**
   * @brief      Provides a random Transfer function out of the ones in the argument
   *
   * @param[in]  range  The range of transfer functions to be given back
   *
   * @return     A random Transfer function according to the given range
   */
  static Transfer_functions next(std::vector<Transfer_functions> range);

  /**
   * @brief      Provides the average range of the given Transfer functions output
   *
   * @param[in]  function  The transfer function in question
   *
   * @return     The average output range.
   */
  static sdouble32 get_average_output_range(Transfer_functions function);

  /**
   * @brief      Apply the given transfer function to the given data
   *
   * @param[in]  function  The function to apply
   * @param[in]  data      The data to apply it to
   *
   * @return     The result of data.
   */
  sdouble32 get_value(Transfer_functions function, sdouble32 data) const;

  /**
   * @brief      Gets a functions derivative calculated form the given data
   *
   * @param[in]  function  The function to use
   * @param[in]  data      The data to use
   *
   * @return     The derivative from data.
   */
  sdouble32 get_derivative(Transfer_functions function, sdouble32 data) const;

  #if(RAFKO_USES_OPENCL)
  std::string get_cl_function_for(Transfer_functions function, std::string x);
  #endif/*(RAFKO_USES_OPENCL)*/

private:
  rafko_mainframe::RafkoSettings& settings;
};

} /* namespace rafko_net */
#endif /* TRANSFER_FUNCTION_H */

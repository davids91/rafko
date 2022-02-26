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

#include <set>
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
  constexpr TransferFunction(const rafko_mainframe::RafkoSettings& settings)
  : settings(settings)
  { }

  /**
   * @brief      Gives a random Transfer Function
   *
   * @return     A random Transfer Function
   */
  static Transfer_functions next(){
    return next({
      transfer_function_identity,
      transfer_function_sigmoid,
      transfer_function_tanh,
      transfer_function_elu,
      transfer_function_selu,
      transfer_function_relu
    });
  }

  /**
   * @brief      Provides a random Transfer function out of the ones in the argument
   *
   * @param[in]  range  The range of transfer functions to be given back
   *
   * @return     A random Transfer function according to the given range
   */
  static Transfer_functions next(std::set<Transfer_functions> range);

  /**
   * @brief      Provides the average range of the given Transfer functions output
   *
   * @param[in]  function  The transfer function in question
   *
   * @return     The average output range.
   */
  static constexpr double get_average_output_range(Transfer_functions function){
    switch(function){
    case transfer_function_sigmoid:
    case transfer_function_tanh:
      return (1.0);
    case transfer_function_elu:
    case transfer_function_relu:
    case transfer_function_selu:
    case transfer_function_identity:
    default:
      return (50.0); /* The averagest number there is */
    }
  }

  /**
   * @brief      Apply the given transfer function to the given data
   *
   * @param[in]  function  The function to apply
   * @param[in]  data      The data to apply it to
   *
   * @return     The result of data.
   */
  double get_value(Transfer_functions function, double data) const;

  /**
   * @brief      Gets a functions derivative calculated form the given data
   *
   * @param[in]  function  The function to use
   * @param[in]  data      The data to use
   *
   * @return     The derivative from data.
   */
  double get_derivative(Transfer_functions function, double data) const;

  #if(RAFKO_USES_OPENCL)
  /**
   * @brief     Generates GPU kernel function code for the provided parameters
   *
   * @param[in]   function    The Transfer function to base the generated kernel code on
   * @param[in]   x           The value on which the transfer function is called upon
   *
   * @return    The generated Kernel code calling the asked transfer function on the parameter
   */
  std::string get_cl_function_for(Transfer_functions function, std::string x);
  #endif/*(RAFKO_USES_OPENCL)*/

private:
  const rafko_mainframe::RafkoSettings& settings;
};

} /* namespace rafko_net */
#endif /* TRANSFER_FUNCTION_H */

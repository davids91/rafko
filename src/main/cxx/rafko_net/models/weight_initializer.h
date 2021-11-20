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

#ifndef WEIGHT_INITIALIZER_H
#define WEIGHT_INITIALIZER_H


#include "rafko_global.h"

#include <cmath>

#include "rafko_mainframe/models/rafko_service_context.h"

namespace rafko_net {

using std::min;
using std::max;
using std::numeric_limits;

using rafko_mainframe::RafkoServiceContext;

class RAFKO_FULL_EXPORT WeightInitializer{
public:
  /**
   * @brief      Constructs the object.
   */
  WeightInitializer(RafkoServiceContext& service_context) noexcept
  : context(service_context)
  { };

  /**
   * @brief      Calculate a weight which fits the Neuron the most based on the configuration parameters
   *             The basis of the number is the transfer function given in the function argument
   *
   * @param[in]  used_transfer_function  The used transfer function
   *
   * @return     The Calculated weight
   */
  virtual sdouble32 next_weight_for(Transfer_functions used_transfer_function) const = 0;

  /**
   * @brief      Calculate a number which fits the Neuron the most based on the configuration parameters
   *
   * @return     The Calculated Memory ratio
   */
  virtual sdouble32 next_memory_filter() const = 0;

  /**
   * @brief      Calculate a bias which fits the Neuron the most based on the configuration parameters
   *
   * @return     The Calculated Bias value
   */
  virtual sdouble32 next_bias() const = 0;

  /**
   * @brief      Sets the functions expected parameters
   *
   * @param[in]  expected_input_number             The exponent input number
   * @param[in]  expected_input_maximum_value_     The exponent input maximum
   */
  void set(uint32 expected_input_number_, sdouble32 expected_input_maximum_value_){
    expected_input_number = max(1u,expected_input_number_);
    if( /* Primitive check if the given number causes overflow or not */
      (numeric_limits<sdouble32>::max() > (expected_input_number_ * std::abs(expected_input_maximum_value_)))
    ){
      expected_input_maximum_value = expected_input_maximum_value_;
    }else if(double_literal(0.0) == expected_input_maximum_value_){
      expected_input_maximum_value = numeric_limits<sdouble32>::epsilon();
    }else{ /* Overflow! Use maximum value */
      expected_input_maximum_value = numeric_limits<sdouble32>::max() / expected_input_number_;
    }
  }

  /**
   * @brief      Calculate a weight which fits the Neuron the most based on the configuration parameters
   *             The basis of the number is the TransferFunction::transfer_function_identity
   *
   * @return     The Calculated Weight value
   */
  sdouble32 next_weight() const{
    return next_weight_for(transfer_function_identity);
  }

  virtual ~WeightInitializer(void) = default;

protected:
  RafkoServiceContext& context;

  /**
   * @brief      Limits the given weight into the limits used in the Neural Network
   *
   * @param[in]  weight  The weight
   *
   * @return     Limited value
   */
  sdouble32 limit_weight(sdouble32 weight) const{
    return min(double_literal(1.0),max(-double_literal(1.0),weight));
  }

  /**
   * Number of estimated @Neuron inputs expected
   */
  uint32 expected_input_number = 0;

  /**
   * Estimated Maximum value of one @Neuron input
   */
  sdouble32 expected_input_maximum_value = numeric_limits<sdouble32>::epsilon();
};

} /* namespace rafko_net */
#endif /* WEIGHT_INITIALIZER_H */

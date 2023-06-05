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

#include "rafko_global.hpp"

#include <math.h>

#include "rafko_mainframe/models/rafko_settings.hpp"

namespace rafko_net {

class RAFKO_EXPORT WeightInitializer {
public:
  /**
   * @brief      Constructs the object.
   */
  constexpr WeightInitializer(
      const rafko_mainframe::RafkoSettings &settings) noexcept
      : m_settings(settings){};

  /**
   * @brief      Calculate a weight which fits the Neuron the most based on the
   * configuration parameters The basis of the number is the transfer function
   * given in the function argument
   *
   * @param[in]  used_transfer_function  The used transfer function
   *
   * @return     The Calculated weight
   */
  virtual double
  next_weight_for(Transfer_functions used_transfer_function) const = 0;

  /**
   * @brief      Calculate a number which fits the Neuron the most based on the
   * configuration parameters
   *
   * @return     The Calculated Memory ratio
   */
  virtual double next_memory_filter() const = 0;

  /**
   * @brief      Calculate a bias which fits the Neuron the most based on the
   * configuration parameters
   *
   * @return     The Calculated Bias value
   */
  virtual double next_bias() const = 0;

  /**
   * @brief      Sets the functions expected parameters
   *
   * @param[in]  expected_input_number             The exponent input number
   * @param[in]  expected_input_maximum_value     The exponent input maximum
   */
  virtual void set(std::uint32_t expected_input_number,
                   double expected_input_maximum_value) {
    m_expectedInputNumber = std::max(1u, expected_input_number);
    if (/* Primitive check if the given number causes overflow or not */
        (std::numeric_limits<double>::max() >
         (expected_input_number * std::abs(expected_input_maximum_value)))) {
      m_expectedInputMaximumValue = expected_input_maximum_value;
    } else if ((0.0) == expected_input_maximum_value) {
      m_expectedInputMaximumValue = std::numeric_limits<double>::epsilon();
    } else { /* Overflow! Use maximum value */
      m_expectedInputMaximumValue =
          std::numeric_limits<double>::max() / expected_input_number;
    }
  }

  /**
   * @brief      Calculate a weight which fits the Neuron the most based on the
   * configuration parameters The basis of the number is the
   * TransferFunction::transfer_function_identity
   *
   * @return     The Calculated Weight value
   */
  double next_weight() const {
    return next_weight_for(transfer_function_identity);
  }

  virtual ~WeightInitializer() = default;

protected:
  const rafko_mainframe::RafkoSettings &m_settings;

  /**
   * Number of estimated @Neuron inputs expected
   */
  std::uint32_t m_expectedInputNumber = 0;

  /**
   * Estimated Maximum value of one @Neuron input
   */
  double m_expectedInputMaximumValue = std::numeric_limits<double>::epsilon();

  /**
   * @brief      Limits the given weight into the limits used in the Neural
   * Network
   *
   * @param[in]  weight  The weight
   *
   * @return     Limited value
   */
  constexpr double limit_weight(double weight) const {
    return std::min((1.0), std::max(-(1.0), weight));
  }
};

} /* namespace rafko_net */
#endif /* WEIGHT_INITIALIZER_H */

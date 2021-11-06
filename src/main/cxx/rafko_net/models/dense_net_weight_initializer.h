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

#ifndef DenseNetWeightInitializer_H
#define DenseNetWeightInitializer_H

#include "rafko_net/models/weight_initializer.h"

namespace rafko_net {

/**
 * @brief      Class for dense net weight initializer. The member functions of the class are documented
 *             under @WeightInitializer. The Aim of this specialization is to provide correct weight,
 *             bias and memory filter initalization to Fully Connected(Dense) Feedforward Neural networks.
 */
class DenseNetWeightInitializer : public WeightInitializer{
public:

  /**
   * @brief      Constructs the object and calls the srand function with the given arguments.
   *             To srand with time(nullptr), the constructor needs to be called with
   *             a true boolean argument or given a seed value.
   */
  DenseNetWeightInitializer(bool seed, ServiceContext& service_context)
  :  WeightInitializer(service_context)
  { if(seed)srand(static_cast<uint32>(time(nullptr))); }

  DenseNetWeightInitializer(
    ServiceContext& service_context, sdouble32 memRatioMin = double_literal(0.0), sdouble32 memRatioMax = double_literal(1.0)
  ): WeightInitializer(service_context)
  {
    memMin = max(double_literal(0.0), min(double_literal(1.0), memRatioMin));
    memMax = min(double_literal(1.0), max(memMin,memRatioMax));
  }

  DenseNetWeightInitializer(
    uint32 seed, ServiceContext& service_context,
    sdouble32 memRatioMin = double_literal(0.0), sdouble32 memRatioMax = double_literal(0.0)
  ): DenseNetWeightInitializer(service_context, memRatioMin, memRatioMax)
  { srand(seed); }

  /**
   * @brief      Configuration functions
   */
  void set(uint32 expected_input_number, sdouble32 expected_input_maximum_value_);
  sdouble32 next_weight_for(TransferFunctions used_transfer_function) const;
  sdouble32 next_memory_filter() const;
  sdouble32 next_bias() const;

private:
  sdouble32 memMin = double_literal(0.0);
  sdouble32 memMax = double_literal(1.0);

  /**
   * @brief      Gets the expected amplitude for a weight with the given transfer function
   *             and configured parameters. Should always be a positive number above
   *             @TransferFunction::epsilon
   *
   * @param[in]  used_transfer_function  The basis transfer function
   *
   * @return     Expected weight amplitude
   */
  sdouble32 get_weight_amplitude(TransferFunctions used_transfer_function) const;

};

} /* namespace rafko_net */

#endif /* DenseNetWeightInitializer_H */

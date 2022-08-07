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

#ifndef DENSE_NET_WEIGHT_INITIALIZER_H
#define DENSE_NET_WEIGHT_INITIALIZER_H

#include <time.h>

#include <math.h>
#include <cstdlib>
#include <algorithm>

#include "rafko_net/models/weight_initializer.h"
#include "rafko_net/models/transfer_function.h"


namespace rafko_net {

/**
 * @brief      Class for dense net weight initializer. The member functions of the class are documented
 *             under @WeightInitializer. The Aim of this specialization is to provide correct weight,
 *             bias and memory filter initalization to Fully Connected(Dense) Feedforward Neural networks.
 */
class RAFKO_FULL_EXPORT DenseNetWeightInitializer : public WeightInitializer{
public:
  /**
   * @brief      Constructs the object and calls the srand function with the given arguments.
   *             To srand with time(nullptr), the constructor needs to be called with
   *             a true boolean argument or given a seed value.
   */
  DenseNetWeightInitializer(
    bool to_seed, const rafko_mainframe::RafkoSettings& settings,
    double memRatioMin = 0.0, double memRatioMax = 1.0
  )
  : DenseNetWeightInitializer(settings, memRatioMin, memRatioMax)
  {
    if(to_seed)srand(static_cast<std::uint32_t>(time(nullptr)));
  }

  DenseNetWeightInitializer(
    const rafko_mainframe::RafkoSettings& settings,
    double memRatioMin = 0.0, double memRatioMax = 1.0
  ): WeightInitializer(settings)
  , memMin(memRatioMin)
  , memMax(std::max(memMin,memRatioMax))
  {
  }

  DenseNetWeightInitializer(
    std::uint32_t seed, const rafko_mainframe::RafkoSettings& settings,
    double memRatioMin = 0.0, double memRatioMax = 1.0
  ): DenseNetWeightInitializer(settings, memRatioMin, memRatioMax)
  {
    srand(seed);
  }

  ~DenseNetWeightInitializer() = default;

  /**
   * @brief      Configuration functions
   */
  using WeightInitializer::set;
  double next_weight_for(Transfer_functions used_transfer_function) const override{
    return ((rand()%2 == 0)?-(1.0):(1.0)) * limit_weight(
      (static_cast<double>(rand())/(static_cast<double>(RAND_MAX/get_weight_amplitude(used_transfer_function))))
    );
  }

  double next_memory_filter() const override{
    if(memMin <  memMax){
      double diff = memMax - memMin;
      return ((0.0) == diff)?0:(
         memMin + (static_cast<double>(rand())/(static_cast<double>(RAND_MAX/diff)))
      );
    } else return memMin;
  }

  double next_bias() const override{
    double amplitude = settings.get_zetta(); /* non-zero value to make ReLU and friends fire right away in training */
    return ( (amplitude / -2.0) + (static_cast<double>(rand()%101)/100.0) * amplitude );
  }

private:
  double memMin;
  double memMax;

  /**
   * @brief      Gets the expected amplitude for a weight with the given transfer function
   *             and configured parameters. Should always be a positive number above
   *             @TransferFunction::epsilon
   *
   * @param[in]  used_transfer_function  The basis transfer function
   *
   * @return     Expected weight amplitude
   */
  double get_weight_amplitude(Transfer_functions used_transfer_function) const{
    double amplitude;
    switch(used_transfer_function){
    case transfer_function_elu:
    case transfer_function_relu:
    case transfer_function_selu:
      amplitude = (sqrt(2 / (expected_input_number))); /* Kaiming initialization */
      break;
    default:
      amplitude = (sqrt(2 / (expected_input_number * expected_input_maximum_value)));
      break;
    }
    return std::max(settings.get_epsilon(),amplitude);
  }
};

} /* namespace rafko_net */

#endif /* DENSE_NET_WEIGHT_INITIALIZER_H */

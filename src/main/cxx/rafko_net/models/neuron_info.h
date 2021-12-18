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

#ifndef NEURON_INFO_H
#define NEURON_INFO_H

#include "rafko_global.h"
#include "rafko_protocol/rafko_net.pb.h"

namespace rafko_net{

class RAFKO_FULL_EXPORT NeuronInfo{
public:

  /**
   * @brief      Gets a neurons estimated size in bytes.
   *
   * @param[in]  neuron  The neuron
   *
   * @return     The neuron estimated size in bytes.
   */
  static uint32 get_neuron_estimated_size_bytes(const Neuron& neuron){
    return (
      neuron.input_weights_size()  * 2/* Byte */ * 2/* fields( interval_size and starts) */
      + neuron.input_indices_size() * 2/* Byte */ * 2/* fields( interval_size and starts) */
    );
  }

  /**
   * @brief      Gets a neurons estimated size in megabytes.
   *
   * @param[in]  neuron  The neuron
   *
   * @return     The neuron estimated size in megabytes.
   */
  static sdouble32 get_neuron_estimated_size_megabytes(const Neuron& neuron){
    return (
      static_cast<sdouble32>(get_neuron_estimated_size_bytes(neuron)) / (double_literal(1024.0) * double_literal(1024.0))
    );
  }

  /**
   * @brief      Determines whether the specified neuron is valid, but does
   *             not take RafkoNet integrity into account (eg.: it doesn't check index validities)
   *
   * @param[in]  neuron  The neuron reference
   */
  static bool is_neuron_valid(const Neuron& neuron);

  /**
   * @brief      Determines whether the given feature is relevant to calculating ( solving ) the neural network
   *
   * @param[in]  feature  The feature to be considered
   */
  static bool is_feature_relevant_to_solution(Neuron_group_features feature);

  /**
   * @brief      Determines whether the given feature is relevant to the error / fitness value of the neural network
   *
   * @param[in]  feature  The feature to be considered
   */
  static bool is_feature_relevant_to_performance(Neuron_group_features feature); /* as in fitness / error */

  /**
   * @brief      Determines whether the given feature is relevant to the training of the neural network
   *
   * @param[in]  feature  The feature to be considered
   */
  static bool is_feature_relevant_to_training(Neuron_group_features feature);
};

} /* namespace rafko_net */

#endif /* NEURON_INFO_H */

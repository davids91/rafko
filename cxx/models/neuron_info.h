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

#include "sparse_net_global.h"
#include "gen/sparse_net.pb.h"

namespace sparse_net_library{

class Neuron_info{
public:

  /**
   * @brief      Gets a neurons estimated size in bytes.
   *
   * @param[in]  neuron  The neuron
   *
   * @return     The neuron estimated size in bytes.
   */
  static uint32 get_neuron_estimated_size_bytes(const Neuron& neuron);

  /**
   * @brief      Determines whether the specified neuron is valid, but does
   *             not take SparseNet integrity into account (eg.: it doesn't check index validities)
   *
   * @param[in]  neuron  The neuron reference
   */
  static bool is_neuron_valid(const Neuron& neuron);
};

} /* namespace sparse_net_library */

#endif /* NEURON_INFO_H */

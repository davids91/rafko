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

#ifndef RAFKO_BACKPROP_NETWORK_INPUT_OPERATION_H
#define RAFKO_BACKPROP_NETWORK_INPUT_OPERATION_H

#include "rafko_global.h"

#include <vector>
#include <memory>
#include <utility>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/services/rafko_assertion_logger.h"

#include "rafko_gym/models/rafko_backpropagation_operation.h"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_FULL_EXPORT RafkoBackpropNetworkInputOperation{
public:
  RafkoBackpropNetworkInputOperation(
    RafkoBackPropagation& queue, rafko_net::RafkoNet& network,
    std::uint32_t input_index_, std::uint32_t weight_index_
  ):RafkoBackpropagationOperation(queue, operations)
  , input_index(input_index_)
  , weight_index(weight_index_)
  {
  }

  void upload_dependencies_to_operations(){
    /*!Note: Network inputs have no dependencies! */
    set_registered();
  }

  void calculate(
    std::uint32 d_w_index, std::uint32 run_index,
    const std::vector<double>& error_data, const std::vector<double>& label_data,
    const DataRingbuffer& neuron_data, const std::vector<double>& network_input,
    const std::vector<double>& spike_function_input
  ){
    value = network_input[input_index] * network.neuron_array(weight_index);
    if(d_w_index == weight_index){
      derivative_value = network_input[input_index];
    }else{
      derivative_value = 0.0;
    }
    set_processed();
  }
private:
  std::uint32_t input_index;
  std::uint32_t weight_index;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_NETWORK_INPUT_OPERATION_H */

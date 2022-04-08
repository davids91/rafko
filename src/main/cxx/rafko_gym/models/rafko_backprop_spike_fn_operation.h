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

#ifndef RAFKO_BACKPROP_SPIKE_FN_OPERATION_H
#define RAFKO_BACKPROP_SPIKE_FN_OPERATION_H

#include "rafko_global.h"

#include <vector>
#include <memory>
#include <utility>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/services/rafko_assertion_logger.h"
#include "rafko_net/models/spike_function.h"

#include "rafko_gym/models/rafko_backpropagation_operation.h"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_FULL_EXPORT RafkoBackpropSpikeFnOperation{
public:
  RafkoBackpropNetworkInputOperation(
    rafko_net::RafkoNet& network, std::uint32_t neuron_index_,
    std::vector<std::unique_ptr<RafkoBackpropagationOperation>>& operations
  ):RafkoBackpropagationOperation(queue, network)
  , neuron_index(neuron_index_)
  {
  }

  void upload_dependencies_to_operations(){
     = push_dependency( /* current value provided by the transfer function */
      ad_operation_neuron_transfer_d, queue, network, neuron_index
    );
     = push_dependency( /* previous value of the neuron */
      ad_operation_neuron_spike_d, queue, network,
      operation_index, network.neuron_array(neuron_index).transfer_function(), 1u /*past_index*/
    );
    set_registered()
  }

  void calculate(
    std::uint32 d_w_index, std::uint32 run_index,
    const std::vector<double>& error_data, const std::vector<double>& label_data,
    const DataRingbuffer& neuron_data, const std::vector<double>& network_input,
    const std::vector<double>& spike_function_input
  ){
    derivative_value = SpikeFunction::get_derivative(
      network.neuron_array(neuron_index).spike_function()
    );
    set_processed()
  }
private:
  const std::uint32_t neuron_index;
  std::shared_ptr<RafkoBackpropagationOperation> ;
  std::shared_ptr<RafkoBackpropagationOperation> ;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_SPIKE_FN_OPERATION_H */

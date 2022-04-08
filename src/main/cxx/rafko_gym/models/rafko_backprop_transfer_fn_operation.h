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

#ifndef RAFKO_BACKPROP_TRANSFER_FN_OPERATION_H
#define RAFKO_BACKPROP_TRANSFER_FN_OPERATION_H

#include "rafko_global.h"

#include <vector>
#include <memory>
#include <utility>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/services/rafko_assertion_logger.h"
#include "rafko_net/models/transfer_function_function.h"

#include "rafko_gym/models/rafko_backpropagation_operation.h"
#include "rafko_gym/models/rafko_backprop_neuron_input_operation.h"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_FULL_EXPORT RafkoBackpropTransferFnOperation{
public:
  RafkoBackpropTransferFnOperation(
    RafkoBackPropagation& queue, rafko_net::RafkoNet& network,
    std::uint32_t neuron_index_, std::shared_ptr<double> tmp_value_buffer_ = {}
  ):RafkoBackpropagationOperation(queue, network)
  {
  }

  void upload_dependencies_to_operations(){
    needed_input_dependency = std::reinterpret_pointer_cast<RafkoBackpropNeuronInputOperation>(
      queue.push_dependency(ad_operation_neuron_input_d, queue, network, neuron_index, 0u/*neuron_input_index*/)
    );
    /*!Note: The first input of the Neuron is to calculate the whole derivative of the Neuron input */
    set_registered();
  }

  void calculate(
    std::uint32 d_w_index, std::uint32 run_index,
    const std::vector<double>& error_data, const std::vector<double>& label_data,
    const DataRingbuffer& neuron_data, const std::vector<double>& network_input,
    const std::vector<double>& spike_function_input
  ){
    RFASSERT(are_dependencies_registered());
    RFASSERT(needed_input_dependency->is_processed());
    value = TransferFunction::get_value(
      network.neuron_array(neuron_index).transfer_function(), needed_input_dependency->get_value()
    );
    derivative_value = TransferFunction::get_derivative( /* d t(f(w))/dx = f'(w) * t'(f(w))*/
      network.neuron_array(neuron_index).transfer_function(),
      needed_input_dependency->get_value(),
      needed_input_dependency->operator();
    );
    set_processed();
  }

private:
  const std::uint32_t neuron_index;
  std::shared_ptr<RafkoBackpropNeuronInputOperation> needed_input_dependency;
  double transfer_function_output;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_TRANSFER_FN_OPERATION_H */

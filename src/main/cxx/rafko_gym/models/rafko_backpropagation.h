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

#ifndef RAFKO_BACKPROPAGATION_H
#define RAFKO_BACKPROPAGATION_H

#include "rafko_global.h"

#include "rafko_gym/models/rafko_backpropagation_operation.h"
#include "rafko_gym/models/rafko_backprop_network_input_operation.h"
#include "rafko_gym/models/rafko_backprop_neuron_input_operation.h"
#include "rafko_gym/models/rafko_backprop_transfer_fn_operation.h"
#include "rafko_gym/models/rafko_backprop_spike_fn_operation.h"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_FULL_EXPORT RafkoBackPropagation{
public:
  RafkoBackPropagation(rafko_net::RafkoNet& network)
  : network(network_)
  {
    network_input_derivatives.reserve(network.input_data_size())
  }

  void build(){
    //TODO: push up operations for every output
    //TODO: loop until all operations are built up
  }

  //TODO: Make threadsafe with mutex maybe?
  template<typename Args...>
  std::shared_ptr<RafkoBackpropagationOperation> push_dependency(Autodiff_operations type, Args... arguments){
    switch(type){
      case ad_operation_objective_d: //TODO:
      case ad_operation_neuron_spike_d:
      case ad_operation_neuron_transfer_d:
        return operations.emplace_back(std::make_shared<RafkoBackpropTransferFnOperation>(arguments...));
      case ad_operation_neuron_input_d:
        return operations.emplace_back(std::make_shared<RafkoBackpropNeuronInputOperation>(arguments...));
      case ad_operation_network_input_d:
        return operations.emplace_back(std::make_shared<RafkoBackpropNetworkInputOperation>(arguments...));
      }break;
    }
  }

private:
  rafko_net::RafkoNet& network;
  std::vector<std::shared_ptr<RafkoBackpropagationOperation>>& operations;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROPAGATION_H */

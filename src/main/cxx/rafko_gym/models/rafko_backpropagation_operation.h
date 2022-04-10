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

#ifndef RAFKO_BACKPROPAGATION_OPERATION_H
#define RAFKO_BACKPROPAGATION_OPERATION_H

#include "rafko_global.h"

#include <memory>
#include <vector>

#include "rafko_protocol/rafko_net.pb.h"

namespace rafko_gym{

class RafkoBackPropagation;
/**
 * @brief A class representing an operation inside the backpropagation logic of reverse mode autodiff.
 * each opeartion is collected with the help of the components of a Neuron ( input-, transfer- and spike function )
 * and objective in each of the classes that inherit from this class; While this class provides a common base
 * which aims to eliminate the stack restrictions present in recursive solutions, by storing every operation
 * in a vector, and providing the chance to upload the operation dependencies into the vector when prompted.
 */
class RAFKO_FULL_EXPORT RafkoBackpropagationOperation{
public:
  RafkoBackpropagationOperation(
    RafkoBackPropagation& queue_, const rafko_net::RafkoNet& network_, std::uint32_t past_index
  ):network(network_)
  , queue(queue_)
  , past_index(past_index_)
  {
    RFASSERT(past_index < network.memory_size());
  }

  virtual void upload_dependencies_to_operations() = 0;

  virtual void calculate(
    std::uint32 d_w_index, std::uint32 run_index,
    const std::vector<std::vector<double>>& network_input, const std::vector<std::vector<double>>& label_data
  ) = 0;

  // std::string get_kernel_function(){
  //
  // }

  double get_derivative(){
    if(!processed)calculate();
    return derivative_value;
  }

  double get_value(){
    if(!processed)calculate();
    return value;
  }

  bool constexpr are_dependencies_registered() const{
    return dependencies_registered;
  }

  bool constexpr is_processed() const{
    return processed;
  }

protected:
  RafkoBackPropagation& queue;
  const rafko_net::RafkoNet& network;
  const std::uint32_t past_index;

  bool processed = false;
  bool dependencies_registered = false;
  double derivative_value = 0.0;
  double value = 0.0;

  void constexpr reset_value(){
    if(processed){
      for(std::unique_ptr<RafkoBackpropagationOperation> dependency : dependencies)
        dependecy->reset_value();
    }
    processed = false;
  }

  void constexpr set_processed(){
    processed = true;
  }

  void constexpr set_registered(){
    dependencies_registered = true;
  }
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROPAGATION_OPERATION_H */

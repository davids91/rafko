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

#ifndef RAFKO_BACKPROPAGATION_DATA_H
#define RAFKO_BACKPROPAGATION_DATA_H

#include "rafko_global.h"

#include <vector>
#include <algorithm>

#include "rafko_utilities/models/data_ringbuffer.h"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_FULL_EXPORT RafkoBackPropagationData{
  using BackpropDerivativeBuffer = rafko_utilities::DataRingbuffer<std::vector<std::vector<double>>>;
  using BackPropValueBuffer = rafko_utilities::DataRingbuffer<>;
public:
  RafkoBackPropagationData(const rafko_net::RafkoNet& network)
  : memory_slots(network.memory_size() + 1u) /* The network always remembers the previous value because of the Spike function */
  , weight_table_size(network.weight_table_size())
  , calculated_derivatives()
  , calculated_values()
  {
  }

  void build(std::uint32_t number_of_operations){
    calculated_values = std::make_unique<BackPropValueBuffer>(
      memory_slots, [number_of_operations](std::vector<double>& element){
        element.resize(number_of_operations);
      }
    );
    calculated_derivatives = std::make_unique<BackpropDerivativeBuffer>(
      memory_slots, [this, number_of_operations](std::vector<std::vector<double>>& element){
        element = std::vector<std::vector<double>>(
          number_of_operations, std::vector<double>(weight_table_size)
        );
      }
    );
    built = true;
  }

  void reset(){
    calculated_values->reset();
    calculated_derivatives->reset();
  }

  void step(){ /*!Note: Not using @clean_step, but only because both the value and derivative will be overwritten anyway.. */
    calculated_values->shallow_step();
    calculated_derivatives->shallow_step();
  }

  void set_value(std::uint32_t operation_index, double value){
    RFASSERT(built);
    RFASSERT(operation_index < calculated_values->get_element(0).size());
    calculated_values->get_element(0u/*past_index*/, operation_index) = value;
  }

  void set_derivative(std::uint32_t operation_index, std::uint32_t d_w_index, double value){
    RFASSERT(built);
    RFASSERT(operation_index < calculated_derivatives->get_element(0u/*past_index*/).size());
    RFASSERT(d_w_index < calculated_derivatives->get_element(0u/*past_index*/, operation_index).size());
    calculated_derivatives->get_element(0u/*past_index*/, operation_index)[d_w_index] = value;
  }

  const BackPropValueBuffer& get_value(){
    return *calculated_values;
  }

  double get_value(std::uint32_t past_index, std::uint32_t operation_index){
    RFASSERT(built);
    if(calculated_values->get_sequence_size() <= past_index) return 0.0;
    RFASSERT(operation_index < calculated_values->get_element(0).size());
    return calculated_values->get_element(past_index, operation_index);
  }

  const BackpropDerivativeBuffer& get_derivative(){
    return *calculated_derivatives;
  }

  double get_derivative(std::uint32_t past_index, std::uint32_t operation_index, std::uint32_t weight_index){
    RFASSERT(built);
    if(calculated_derivatives->get_sequence_size() <= past_index) return 0.0;
    RFASSERT(operation_index < calculated_derivatives->get_element(0).size());
    RFASSERT(weight_index < calculated_derivatives->get_element(past_index, operation_index).size());
    return calculated_derivatives->get_element(past_index, operation_index)[weight_index];
  }

private:
  const std::uint32_t memory_slots;
  const std::uint32_t weight_table_size;
  std::unique_ptr<BackpropDerivativeBuffer> calculated_derivatives; /* {runs, operations, d_w values} */
  std::unique_ptr<BackPropValueBuffer> calculated_values; /* {runs, operations} */
  bool built = false;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROPAGATION_DATA_H */

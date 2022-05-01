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
class RAFKO_FULL_EXPORT RafkoBackpropagationData{

  /* For every run the network remembers the calcuated result is stored in a type like this */
  using NetworkValueBuffer = rafko_utilities::DataRingbuffer<>;

  /* For every run the network remembers, the per weight derivative value of every operation  is stored */
  using NetworkDerivativeBuffer = rafko_utilities::DataRingbuffer<std::vector<std::vector<double>>>;

  /* For every sequence */
  using SequenceDerivativeBuffer = rafko_utilities::DataRingbuffer<>;

public:
  RafkoBackpropagationData(const rafko_net::RafkoNet& network)
  : memory_slots(network.memory_size() + 1u) /* The network always remembers the previous value because of the Spike function */
  , weight_table_size(network.weight_table_size())
  , weight_relevant_operation_count(0u)
  , calculated_derivatives()
  , calculated_values()
  , sequence_derivatives()
  {
  }

  /**
   * @brief   Constructs ( or re-constructs ) the buffers based on the provided information
   *
   * @param[in]     number_of_operations        The number of backpropagation operations to store inside the buffers
   * @param[in]     relevant_operation_count    The number of backpropagation operations to relevant to weights, i.e. not only used internally
   * @param[in]     sequence_size               The size of a sequence the network is going to be running in
   */
  void build(std::uint32_t number_of_operations, std::uint32_t relevant_operation_count, std::uint32_t sequence_size){
    calculated_values = std::make_unique<NetworkValueBuffer>(
      memory_slots, [number_of_operations](std::vector<double>& element){
        element.resize(number_of_operations);
      }
    );
    calculated_derivatives = std::make_unique<NetworkDerivativeBuffer>(
      memory_slots, [this, number_of_operations](std::vector<std::vector<double>>& element){
        element = std::vector<std::vector<double>>(
          number_of_operations, std::vector<double>(weight_table_size)
        );
      }
    );
    sequence_derivatives = std::make_unique<SequenceDerivativeBuffer>(
      sequence_size, [this](std::vector<double>& element){
        element.resize(weight_table_size);
      }
    );
    built = true;
    weight_relevant_operation_count = relevant_operation_count;
  }


  /**
   * @brief Erases the data stored in the data buffers
   */
  void reset(){
    if(built){
      calculated_values->reset();
      calculated_derivatives->reset();
      sequence_derivatives->reset();
    }
  }

  /**
   * @brief   shifts the iterators inside the buffers one step forward, as if the network is finished with one iteration of calculations.
   *          network values and derivatives now contain "garbage", which is the data from the last iteration the network is
   *          not supposed to remember now, while sequence derivatives are filled with zero values.
   */
  void step(){
    RFASSERT(built);
    /*!Note: Not using @clean_step, but only because both the value and derivative will be overwritten anyway.. */
    calculated_values->shallow_step();
    /* using clean step, because the at each step the values depend on being clean (0.0).. */
    calculated_derivatives->clean_step(); /* ..so sequence truncation would have 0.0 if sequence is excluded and not calculated */
    sequence_derivatives->clean_step(); /* ..and so the averages would start with 0.0 as initial value */
  }

  /**
   * @brief     Stores the provided value as a result of an operation inside the network for the current iteration of the bufffers
   *
   * @param[in]    operation_index   The index of the operation the value is stored for
   * @param[in]    value             The value to store
   */
  void set_value(std::uint32_t operation_index, double value){
    RFASSERT(built);
    RFASSERT(operation_index < calculated_values->get_element(0).size());
    calculated_values->get_element(0u/*past_index*/, operation_index) = value;
  }

  /**
   * @brief     Stores the provided value as a derivative value of an operation inside the network for the current iteration of the bufffers
   *
   * @param[in]    operation_index   The index of the operation the value is stored for
   * @param[in]    d_w_index         The index of the weight the partial drivative is calculated for
   * @param[in]    value             The derivative value to store
   */
  void set_derivative(std::uint32_t operation_index, std::uint32_t d_w_index, double value){
    RFASSERT(built);
    RFASSERT(operation_index < calculated_derivatives->get_element(0u/*past_index*/).size());
    RFASSERT(d_w_index < calculated_derivatives->get_element(0u/*past_index*/, operation_index).size());
    calculated_derivatives->get_element(0u/*past_index*/, operation_index)[d_w_index] = value;
    if(operation_index < weight_relevant_operation_count){
      /*!Note: The first operations are the objective operations for the outputs, only those matter in this case */
      double& stored_avg = sequence_derivatives->get_element(0u/*past_index*/)[d_w_index];
      stored_avg = (stored_avg + value)/2.0;
    }
  }

  /**
   * @brief provides access to the underlying buffer for the network operation values
   */
  const NetworkValueBuffer& get_value(){
    RFASSERT(built);
    return *calculated_values;
  }

  NetworkValueBuffer& get_mutable_value(){
    return *calculated_values;
  }

  /**
   * @brief     queries the network operation calculated result under the given parameters
   *
   * @param[in]    past_index        The past_index of the iteration the network supposedly remembers
   * @param[in]    operation_index   The index of the operation the value is queried for
   */
  double get_value(std::uint32_t past_index, std::uint32_t operation_index){
    RFASSERT(built);
    if(calculated_values->get_sequence_size() <= past_index) return 0.0;
    RFASSERT(operation_index < calculated_values->get_element(0).size());
    return calculated_values->get_element(past_index, operation_index);
  }

  /**
   * @brief provides access to the underlying buffer for the network operation derivatives
   */
  const NetworkDerivativeBuffer& get_actual_derivative(){
    RFASSERT(built);
    return *calculated_derivatives;
  }

  /**
   * @brief     queries the network operation calculated derivative under the given parameters
   *
   * @param[in]    past_index        The past_index of the iteration the network supposedly remembers
   * @param[in]    operation_index   The index of the operation the value is queried for
   * @param[in]    weight_index      The index of the weight the value is queried for
   */
  double get_derivative(std::uint32_t past_index, std::uint32_t operation_index, std::uint32_t weight_index){
    RFASSERT(built);
    if(calculated_derivatives->get_sequence_size() <= past_index) return 0.0;
    RFASSERT(operation_index < calculated_derivatives->get_element(0).size());
    RFASSERT(weight_index < calculated_derivatives->get_element(past_index, operation_index).size());
    return calculated_derivatives->get_element(past_index, operation_index)[weight_index];
  }

  /**
   * @brief provides access to the underlying buffer for the network operation derivatives
   */
  const NetworkDerivativeBuffer& get_sequence_derivative(){
    RFASSERT(built);
    return *calculated_derivatives;
  }

  /**
   * @brief     queries the calculated derivative for the given sequence and weight
   *
   * @param[in]    past_sequence_index    The index of the previous loop to collect the derivative from
   * @param[in]    weight_index           The index of the weight the value is queried for
   */
  double get_average_derivative(std::uint32_t past_sequence_index, std::uint32_t weight_index){
    RFASSERT(built);
    if(sequence_derivatives->get_sequence_size() <= past_sequence_index) return 0.0;
    RFASSERT(weight_index < sequence_derivatives->get_element(past_sequence_index).size());
    return sequence_derivatives->get_element(past_sequence_index, weight_index);
  }

  /**
   * @brief provides access to the underlying buffer for the weight derivative buffers
   */
  const SequenceDerivativeBuffer& get_average_derivative(){
    RFASSERT(built);
    return *sequence_derivatives;
  }

private:
  const std::uint32_t memory_slots;
  const std::uint32_t weight_table_size;
  std::uint32_t weight_relevant_operation_count;
  std::unique_ptr<NetworkDerivativeBuffer> calculated_derivatives; /* {runs, operations, d_w values} */
  std::unique_ptr<NetworkValueBuffer> calculated_values; /* {runs, operations} */
  std::unique_ptr<SequenceDerivativeBuffer> sequence_derivatives; /* past_sequences_index, average d_w_values */
  //TODO: Maybe don't store just averages of output objective operations?
  bool built = false;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROPAGATION_DATA_H */

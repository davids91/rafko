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

#include "rafko_global.hpp"

#include <algorithm>
#include <vector>

#include "rafko_mainframe/services/rafko_assertion_logger.hpp"
#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_utilities/models/data_ringbuffer.hpp"

namespace rafko_gym {

/**
 * @brief   The data partition to store neural propagation information: values,
 * derivatives for each weight
 */
class RAFKO_EXPORT RafkoBackpropagationData {

  /* For every run the network remembers the calcuated result is stored in a
   * type like this */
  using NetworkValueBuffer = rafko_utilities::DataRingbuffer<>;

  /* For every run the network remembers, the per weight derivative value of
   * every operation  is stored */
  using NetworkDerivativeBuffer =
      rafko_utilities::DataRingbuffer<std::vector<std::vector<double>>>;

  /* For every sequence */
  using SequenceDerivativeBuffer = rafko_utilities::DataRingbuffer<>;

public:
  RafkoBackpropagationData(const rafko_net::RafkoNet &network)
      : m_memorySlots(network.memory_size() +
                      1u) /* The network always remembers the previous value
                             because of the Spike function */
        ,
        m_weightTableSize(network.weight_table_size()),
        m_weightRelevantOperationCount(0u), m_calculatedDerivatives(),
        m_calculatedValues(), m_sequenceDerivatives() {}

  /**
   * @brief   Constructs ( or re-constructs ) the buffers based on the provided
   * information
   *
   * @param[in]     number_of_operations        The number of backpropagation
   * operations to store inside the buffers
   * @param[in]     relevant_operation_count    The number of backpropagation
   * operations to relevant to weights, i.e. not only used internally
   * @param[in]     sequence_size               The size of a sequence the
   * network is going to be running in
   */
  void build(std::uint32_t number_of_operations,
             std::uint32_t relevant_operation_count,
             std::uint32_t sequence_size);

  /**
   * @brief Erases the data stored in the data buffers
   */
  void reset();

  /**
   * @brief   shifts the iterators inside the buffers one step forward, as if
   * the network is finished with one iteration of calculations. network values
   * and derivatives now contain "garbage", which is the data from the last
   * iteration the network is not supposed to remember now, while sequence
   * derivatives are filled with zero values.
   */
  void step();

  /**
   * @brief   sets the switch deciding whether or not sequence_derivatives are
   * updated when setting derivatives
   *
   * @param[in]   update    true, if sequence_derivatives are to be updated when
   * storing derivative calculations
   */
  constexpr void set_weight_derivative_update(bool update) {
    m_updateWeightDerivative = update;
  }

  /**
   * @brief     Stores the provided value as a result of an operation inside the
   * network for the current iteration of the bufffers
   *
   * @param[in]    operation_index   The index of the operation the value is
   * stored for
   * @param[in]    value             The value to store
   */
  void set_value(std::uint32_t operation_index, double value) {
    RFASSERT(m_built);
    RFASSERT(operation_index < m_calculatedValues->get_element(0).size());
    m_calculatedValues->get_element(0u /*past_index*/, operation_index) = value;
  }

  /**
   * @brief     Stores the provided value as a derivative value of an operation
   * inside the network for the current iteration of the bufffers
   *
   * @param[in]    operation_index   The index of the operation the value is
   * stored for
   * @param[in]    d_w_index         The index of the weight the partial
   * drivative is calculated for
   * @param[in]    value             The derivative value to store
   */
  void set_derivative(std::uint32_t operation_index, std::uint32_t d_w_index,
                      double value);

  /**
   * @brief provides const access to the underlying buffer for the network
   * operation values
   */
  const NetworkValueBuffer &get_value() {
    RFASSERT(m_built);
    return *m_calculatedValues;
  }

  /**
   * @brief provides non-const access to the underlying buffer for the network
   * operation values
   */
  NetworkValueBuffer &get_mutable_value() { return *m_calculatedValues; }

  /**
   * @brief     queries the network operation calculated result under the given
   * parameters
   *
   * @param[in]    past_index        The past_index of the iteration the network
   * supposedly remembers
   * @param[in]    operation_index   The index of the operation the value is
   * queried for
   */
  double get_value(std::uint32_t past_index, std::uint32_t operation_index) {
    RFASSERT(m_built);
    if (m_calculatedValues->get_sequence_size() <= past_index)
      return 0.0;
    RFASSERT(operation_index < m_calculatedValues->get_element(0).size());
    return m_calculatedValues->get_element(past_index, operation_index);
  }

  /**
   * @brief provides access to the underlying buffer for the network operation
   * derivatives
   */
  const NetworkDerivativeBuffer &get_actual_derivative() {
    RFASSERT(m_built);
    return *m_calculatedDerivatives;
  }

  /**
   * @brief     queries the network operation calculated derivative under the
   * given parameters
   *
   * @param[in]    past_index        The past_index of the iteration the network
   * supposedly remembers
   * @param[in]    operation_index   The index of the operation the value is
   * queried for
   * @param[in]    weight_index      The index of the weight the value is
   * queried for
   */
  double get_derivative(std::uint32_t past_index, std::uint32_t operation_index,
                        std::uint32_t weight_index) {
    RFASSERT(m_built);
    if (m_calculatedDerivatives->get_sequence_size() <= past_index)
      return 0.0;
    RFASSERT(operation_index < m_calculatedDerivatives->get_element(0).size());
    RFASSERT(weight_index <
             m_calculatedDerivatives->get_element(past_index, operation_index)
                 .size());
    return m_calculatedDerivatives->get_element(past_index,
                                                operation_index)[weight_index];
  }

  /**
   * @brief     provides access to the underlying buffer for the network
   * operation derivatives stored explicitly to base weight updates upon.
   */
  const NetworkDerivativeBuffer &get_sequence_derivative() {
    RFASSERT(m_built);
    return *m_calculatedDerivatives;
  }

  /**
   * @brief     queries the calculated derivative for the given sequence and
   * weight
   *
   * @param[in]    past_sequence_index    The index of the previous loop to
   * collect the derivative from
   * @param[in]    weight_index           The index of the weight the value is
   * queried for
   */
  double get_average_derivative(std::uint32_t past_sequence_index,
                                std::uint32_t weight_index) const {
    RFASSERT(m_built);
    if (m_sequenceDerivatives->get_sequence_size() <= past_sequence_index)
      return 0.0;
    RFASSERT(weight_index <
             m_sequenceDerivatives->get_element(past_sequence_index).size());
    return m_sequenceDerivatives->get_element(past_sequence_index,
                                              weight_index);
  }

  /**
   * @brief provides access to the underlying buffer for the weight derivative
   * buffers
   */
  const SequenceDerivativeBuffer &get_average_derivative() {
    RFASSERT(m_built);
    return *m_sequenceDerivatives;
  }

private:
  const std::uint32_t m_memorySlots;
  const std::uint32_t m_weightTableSize;
  std::uint32_t m_weightRelevantOperationCount;
  std::unique_ptr<NetworkDerivativeBuffer>
      m_calculatedDerivatives; /* {runs, operations, d_w values} */
  std::unique_ptr<NetworkValueBuffer>
      m_calculatedValues; /* {runs, operations} */
  std::unique_ptr<SequenceDerivativeBuffer>
      m_sequenceDerivatives; /* past_sequences_index, average d_w_values */
  bool m_built = false;
  bool m_updateWeightDerivative = true;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROPAGATION_DATA_H */

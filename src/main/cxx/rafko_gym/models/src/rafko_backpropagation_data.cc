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
#include "rafko_gym/models/rafko_backpropagation_data.hpp"

namespace rafko_gym {

void RafkoBackpropagationData::build(std::uint32_t number_of_operations,
                                     std::uint32_t relevant_operation_count,
                                     std::uint32_t sequence_size) {
  m_calculatedValues = std::make_unique<NetworkValueBuffer>(
      m_memorySlots, [number_of_operations](std::vector<double> &element) {
        element.resize(number_of_operations);
      });
  m_calculatedDerivatives = std::make_unique<NetworkDerivativeBuffer>(
      m_memorySlots,
      [this, &number_of_operations](std::vector<std::vector<double>> &element) {
        element = std::vector<std::vector<double>>(
            number_of_operations, std::vector<double>(m_weightTableSize));
      });
  m_sequenceDerivatives = std::make_unique<SequenceDerivativeBuffer>(
      sequence_size, [this](std::vector<double> &element) {
        element.resize(m_weightTableSize);
      });
  m_built = true;
  m_weightRelevantOperationCount = relevant_operation_count;
}

void RafkoBackpropagationData::reset() {
  if (m_built) {
    m_calculatedValues->reset();
    m_calculatedDerivatives->reset();
    m_sequenceDerivatives->reset();
  }
}

void RafkoBackpropagationData::step() {
  RFASSERT(m_built);
  /*!Note: Not using @clean_step, here because the value will be overwritten
   * anyway.. */
  m_calculatedValues->shallow_step();
  /* using clean step, because the at each step the values depend on being
   * clean (0.0).. */
  m_calculatedDerivatives
      ->clean_step(); /* ..so sequence truncation would have 0.0 if sequence
                         is excluded and not calculated */
  m_sequenceDerivatives->clean_step(); /* ..and so the averages would start
                                          with 0.0 as initial value */
}

void RafkoBackpropagationData::set_derivative(std::uint32_t operation_index,
                                              std::uint32_t d_w_index,
                                              double value) {
  RFASSERT(m_built);
  RFASSERT(operation_index <
           m_calculatedDerivatives->get_element(0u /*past_index*/).size());
  RFASSERT(d_w_index < m_calculatedDerivatives
                           ->get_element(0u /*past_index*/, operation_index)
                           .size());
  m_calculatedDerivatives->get_element(0u /*past_index*/,
                                       operation_index)[d_w_index] = value;
  if ((m_updateWeightDerivative) &&
      (operation_index < m_weightRelevantOperationCount)) {
    /*!Note: The first operations are the objective operations for the
     * outputs, only those matter in this case */
    double &stored_avg =
        m_sequenceDerivatives->get_element(0u /*past_index*/)[d_w_index];
    stored_avg = (stored_avg + value) / 2.0;
  }
}

} /* namespace rafko_gym */

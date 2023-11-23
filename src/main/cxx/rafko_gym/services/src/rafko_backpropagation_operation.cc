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
#include "rafko_gym/services/rafko_backpropagation_operation.hpp"

#include "rafko_mainframe/services/rafko_assertion_logger.hpp"

#include <limits>

namespace rafko_gym {

RafkoBackpropagationOperation::RafkoBackpropagationOperation(
    RafkoBackpropagationData &data, const rafko_net::RafkoNet &network,
    std::uint32_t operation_index, Autodiff_operations type)
    : m_data(data), m_network(network), m_operationIndex(operation_index),
      m_type(type) {}

std::uint32_t RafkoBackpropagationOperation::get_max_dependency_index() {
  RFASSERT(are_dependencies_registered());
  std::vector<Dependency> dependencies = get_dependencies();
  for (const Dependency &dep : dependencies)
    RFASSERT(dep->operation_index_finalised());
  auto found_element = std::max_element(
      dependencies.begin(), dependencies.end(),
      [](const Dependency &a, const Dependency &b) {
        return a->get_operation_index() < b->get_operation_index();
      });
  if (found_element == dependencies.end())
    return std::numeric_limits<std::uint32_t>::max();
  else
    return (*found_element)->get_operation_index();
}

double
RafkoBackpropagationOperation::get_derivative(std::uint32_t past_index,
                                              std::uint32_t d_w_index) const {
  return m_data.get_derivative(past_index, get_operation_index(), d_w_index);
}

double
RafkoBackpropagationOperation::get_value(std::uint32_t past_index) const {
  return m_data.get_value(past_index, get_operation_index());
}

void RafkoBackpropagationOperation::set_derivative(std::uint32_t d_w_index,
                                                   double value) {
  m_data.set_derivative(get_operation_index(), d_w_index, value);
}

void RafkoBackpropagationOperation::set_value(double value) {
  m_data.set_value(get_operation_index(), value);
}

} /* namespace rafko_gym */

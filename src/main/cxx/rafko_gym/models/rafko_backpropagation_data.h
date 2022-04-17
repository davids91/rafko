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

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_FULL_EXPORT RafkoBackPropagationData{
public:
  RafkoBackPropagationData(const rafko_net::RafkoNet& network)
  : weight_table_size(network.weight_table_size())
  {
  }

  void build(std::uint32_t number_of_operations){
    for(std::vector<double>& values : calculated_values)
      values = std::vector<double>(number_of_operations);

    for(std::vector<std::vector<double>>& past : calculated_derivatives)
      for(std::vector<double>& operation : past)
        operation = std::vector<double>(weight_table_size);
    built = true;
  }

  void set_value(std::uint32_t run_index, std::uint32_t operation_index, double value){
    RFASSERT(built);
    RFASSERT(run_index < calculated_values.size());
    RFASSERT(operation_index < calculated_values[run_index].size());
    calculated_values[run_index][operation_index] = value;
  }

  // TODO:constexpr these
  void set_derivative(
    std::uint32_t run_index, std::uint32_t operation_index,
    std::uint32_t d_w_index, double value
  ){
    RFASSERT(built);
    RFASSERT(run_index < calculated_values.size());
    RFASSERT(d_w_index < calculated_values[run_index].size());
    calculated_derivatives[run_index][operation_index][d_w_index] = value;
  }

  double get_value(std::uint32_t run_index, std::uint32_t operation_index){
    RFASSERT(built);
    RFASSERT(run_index < calculated_values.size());
    RFASSERT(operation_index < calculated_values[run_index].size());
    return calculated_values[run_index][operation_index];
  }

  double get_derivative(std::uint32_t run_index, std::uint32_t operation_index, std::uint32_t weight_index){
    RFASSERT(built);
    RFASSERT(run_index < calculated_derivatives.size());
    RFASSERT(operation_index < calculated_derivatives[run_index].size());
    RFASSERT(weight_index < calculated_derivatives[run_index][operation_index].size());
    return calculated_derivatives[run_index][operation_index][weight_index];
  }

private:
  const std::uint32_t weight_table_size;
  std::vector<std::vector<std::vector<double>>> calculated_derivatives; /* {runs, operations, d_w values} */
  std::vector<std::vector<double>> calculated_values; /* {runs, operations} */
  bool built = false;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROPAGATION_DATA_H */

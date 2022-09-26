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

#ifndef FUNCTION_FACTORY_H
#define FUNCTION_FACTORY_H

#include "rafko_global.hpp"

#include <memory>
#include <stdexcept>

#include "rafko_protocol/rafko_net.pb.h"

#include "rafko_gym/services/cost_function_squared_error.hpp"
#include "rafko_gym/services/cost_function_mse.hpp"
#include "rafko_gym/services/cost_function_cross_entropy.hpp"
#include "rafko_gym/services/cost_function_binary_cross_entropy.hpp"

namespace rafko_gym{

class RAFKO_EXPORT FunctionFactory{
public:

  /**
   * @brief      Builds a cost function.
   *
   * @param[in]  feature_size   The size of one feature
   * @param[in]  the_function   The cost function to build
   * @param[in]  settings        The service settings
   *
   * @return     The cost function.
   */
  [[nodiscard]] static std::unique_ptr<CostFunction> build_cost_function(Cost_functions the_function, const rafko_mainframe::RafkoSettings& settings){
    switch(the_function){
      case cost_function_mse:                     return std::make_unique<CostFunctionMSE>(settings);
        case cost_function_squared_error:         return std::make_unique<CostFunctionSquaredError>(settings);
        case cost_function_cross_entropy:         return std::make_unique<CostFunctionCrossEntropy>(settings);
        case cost_function_binary_cross_entropy:  return std::make_unique<CostFunctionBinaryCrossEntropy>(settings);
      default: throw std::runtime_error("Unknown cost function requested from builder!");
    }
  }
};

} /* namespace rafko_net */

#endif /* FUNCTION_FACTORY_H */

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

#include "sparse_net_global.h"

#include <memory>
#include <stdexcept>

#include "gen/common.pb.h"
#include "gen/sparse_net.pb.h"
#include "rafko_mainframe/models/service_context.h"

#include "sparse_net_library/models/cost_function_squared_error.h"
#include "sparse_net_library/models/cost_function_mse.h"

namespace sparse_net_library{

using std::unique_ptr;

class Function_factory{
public:
  /**
   * @brief      Builds a cost function.
   *
   * @param[in]  net            The @SparseNet that decides which cost function to build
   * @param[in]  context        The service context
   *
   * @return     The cost function.
   */
  static unique_ptr<Cost_function> build_cost_function(const SparseNet& net, cost_functions the_function, Service_context& context){
    return build_cost_function(net.output_neuron_number(), the_function, context);
  }

  /**
   * @brief      Builds a cost function.
   *
   * @param[in]  feature_size   The size of one feature
   * @param[in]  the_function   The cost function to build
   * @param[in]  context        The service context
   *
   * @return     The cost function.
   */
  static unique_ptr<Cost_function> build_cost_function(uint32 feature_size, cost_functions the_function, Service_context& context){
    switch(the_function){
      case COST_FUNCTION_MSE: 
        return std::make_unique<Cost_function_mse>(feature_size, context);
      case COST_FUNCTION_SQUARED_ERROR: 
        return std::make_unique<Cost_function_squared_error>(feature_size, context);
      default: throw std::runtime_error("Unknown cost function requested from builder!");
    }
  }
};

} /* namespace sparse_net_library */

#endif /* FUNCTION_FACTORY_H */

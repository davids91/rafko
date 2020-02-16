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
 *    along with Foobar.  If not, see <https://www.gnu.org/licenses/> or
 *    <https://github.com/davids91/rafko/blob/master/LICENSE>
 */

#ifndef FUNCTION_FACTORY_H
#define FUNCTION_FACTORY_H

#include "gen/common.pb.h"
#include "gen/sparse_net.pb.h"
#include "models/service_context.h"
#include "models/cost_function_quadratic.h"

#include <memory>

namespace sparse_net_library{

using std::unique_ptr;

class Function_factory{
public:
  static unique_ptr<Cost_function> build_cost_function(
    const SparseNet& net, vector<vector<sdouble32>>& label_samples,
    Service_context context = Service_context()
  ){
    return build_cost_function(net.cost_function(), label_samples, context);
  }
  static unique_ptr<Cost_function> build_cost_function(
    cost_functions the_function, vector<vector<sdouble32>>& label_samples,
    Service_context context = Service_context()
  ){
    switch(the_function){
      case COST_FUNCTION_QUADRATIC: return std::make_unique<Cost_function_quadratic>(label_samples, context);
      default: throw "Unknown cost function requested from builder!";
    }
  }
};

} /* namespace sparse_net_library */

#endif /* FUNCTION_FACTORY_H */

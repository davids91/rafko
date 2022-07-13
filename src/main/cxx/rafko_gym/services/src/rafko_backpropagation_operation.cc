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

namespace rafko_gym{

std::uint32_t RafkoBackpropagationOperation::get_max_dependency_index(){
  RFASSERT(are_dependencies_registered());
  std::vector<Dependency> dependencies = get_dependencies();
  for(const Dependency& dep : dependencies)
    RFASSERT(dep->operation_index_finalised());
  auto found_element = std::max_element(
    dependencies.begin(), dependencies.end(),
    [](const Dependency& a, const Dependency& b){
      return a->get_operation_index() < b->get_operation_index();
    }
  );
  if(found_element == dependencies.end())
    return std::numeric_limits<std::uint32_t>::max();
    else return (*found_element)->get_operation_index();
}

} /* namespace rafko_gym */

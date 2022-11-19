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

#ifndef RAFQ_ENVIRONMENT_H
#define RAFQ_ENVIRONMENT_H

#include "rafko_global.hpp"

#include <vector>

namespace rafko_gym{
/**
 * @brief      This class helps index a state paired with a number of Action Q-value pairs
 */
class RAFKO_EXPORT RafQEnvironment
{
public:
  using DataType = std::vector<double>;
  using MaybeDataType = std::optional<std::reference_wrapper<const DataType>>; 

  RafQEnvironment(std::uint32_t state_size, std::uint32_t action_size)
  : m_stateSize(state_size)
  , m_actionSize(action_size)
  {
  }

  struct StateTransition{
    MaybeDataType m_resultState = {};
    const double m_resultQValue = 0.0;
    const bool m_terminal = true;
  };

  virtual StateTransition next(const DataType& state, const DataType& action) = 0;

  std::uint32_t state_size(){
    return m_stateSize;
  }

  std::uint32_t action_size(){
    return m_actionSize;
  }

private:
  const std::uint32_t m_stateSize;
  const std::uint32_t m_actionSize;
};

} /* namespace rafko_gym */
#endif /* RAFQ_ENVIRONMENT_H */

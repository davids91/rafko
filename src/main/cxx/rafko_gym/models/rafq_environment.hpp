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
#include <optional>

#include "rafko_utilities/models/const_vector_subrange.hpp"

namespace rafko_gym{
/**
 * @brief      This class helps index a state paired with a number of Action Q-value pairs
 */
class RAFKO_EXPORT RafQEnvironment
{
public:
  using DataType = std::vector<double>;
  using DataView = rafko_utilities::ConstVectorSubrange<DataType::const_iterator>;
  using MaybeDataType = std::optional<std::reference_wrapper<const DataType>>; 

  struct EnvProperties{
    const double m_mean = 0.0;
    const double m_standardDeviation = 1.0;
  };

  RafQEnvironment(
    std::uint32_t state_size, std::uint32_t action_size, 
    EnvProperties state_properties = {0.0, 1.0}, EnvProperties action_properties = {0.0, 1.0}
  )
  : m_stateSize(state_size)
  , m_actionSize(action_size)
  , m_stateProperties(state_properties)
  , m_actionProperties(action_properties)
  {
  }

  const EnvProperties& state_peoperties(){
    return m_stateProperties;
  }

  const EnvProperties& action_properties(){
    return m_actionProperties;
  }

  struct StateTransition{
    MaybeDataType m_resultState = {};
    const double m_resultQValue = 0.0;
    const bool m_terminal = true;
  };

  virtual StateTransition next(DataView state, DataView action) = 0;

  constexpr std::uint32_t state_size(){
    return m_stateSize;
  }

  constexpr std::uint32_t action_size(){
    return m_actionSize;
  }

private:
  const std::uint32_t m_stateSize;
  const std::uint32_t m_actionSize;
  const EnvProperties m_stateProperties;
  const EnvProperties m_actionProperties;
};

} /* namespace rafko_gym */
#endif /* RAFQ_ENVIRONMENT_H */

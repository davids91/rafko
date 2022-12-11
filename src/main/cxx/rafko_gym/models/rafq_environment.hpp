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

namespace rafko_gym {


/**
 * @brief      This class is an interface for providing state-action pairsto the qlearning framework
 */
class RAFKO_EXPORT RafQEnvironment
{
public:
  using FeatureVector = std::vector<double>;
  using FeatureView = rafko_utilities::ConstVectorSubrange<FeatureVector::const_iterator>;
  using MaybeFeatureVector = std::optional<std::reference_wrapper<const FeatureVector>>; 

  /**
   * @struct    environmental properties. THey are used to generate new random actions
   */
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

  /**
   * @brief   returns with the stored properties for one state
   * 
   * @return    const reference of the stored environmental properties
   */
  const EnvProperties& state_properties() const{
    return m_stateProperties;
  }

  /**
   * @brief   returns with the stored properties for one action
   * 
   * @return    const reference of the stored environmental properties
   */
  const EnvProperties& action_properties() const{
    return m_actionProperties;
  }

  /**
   * @struct    a structure containing the optional new state, resulting q_value and terminal flag for a state transition
   */
  struct StateTransition{
    MaybeFeatureVector m_resultState = {};
    const double m_resultQValue = 0.0;
    const bool m_terminal = true;
  };

  /**
   * @brief    resets the environment to its initial state
   */
  virtual void reset() = 0;

  /**
   * @brief    provides access to the environments current state
   * 
   * @return    a bufffer reference to the actual state of the environment wrapped in an optional 
   */
  virtual MaybeFeatureVector current_state() const = 0;

  /**
   * @brief    steps the environment forward with the provided action
   * 
   * @param[in]     Buffer to the action the environment should step forward with
   * 
   * @return    An appropriate @StateTransition structure
   */  
  virtual StateTransition next(FeatureView action) = 0;

  /**
   * @brief    provides access to the next step for the given state-action pair
   * 
   * @param[in]     Buffer to the state the environment should be when applying the given action
   * @param[in]     Buffer to the action the environment should step forward with
   * 
   * @return    An appropriate @StateTransition structure
   */
  virtual StateTransition next(FeatureView state, FeatureView action) const = 0;

  /**
   * @brief   provides the size of one state feature vector
   * 
   * @return    size of one state feature vector
   */
  constexpr std::uint32_t state_size() const{
    return m_stateSize;
  }

  /**
   * @brief   provides the size of one action feature vector
   * 
   * @return    size of one action feature vector
   */
  constexpr std::uint32_t action_size() const{
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

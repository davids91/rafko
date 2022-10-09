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

#ifndef RAFKO_AUTONOMOUS_ENTITY_H
#define RAFKO_AUTONOMOUS_ENTITY_H

#include "rafko_global.hpp"

#include <memory>

#include <google/protobuf/arena.h>

#include "rafko_mainframe/models/rafko_settings.hpp"

namespace rafko_mainframe{

/**
 * @brief      An Autonomous entity means an object within te framework with its own scope of settings and Arena.
 *             It's meant to be a long lived object handling other objects with shorter lifetimes
 */
class RAFKO_EXPORT RafkoAutonomousEntity{
public:
  RafkoAutonomousEntity(std::shared_ptr<rafko_mainframe::RafkoSettings> settings = {})
  : m_settings(settings?settings:std::make_shared<rafko_mainframe::RafkoSettings>())
  , m_arena(initialize_arena(*m_settings))
  {
    if(m_arena)m_settings->set_arena_ptr(m_arena.get());
  }

protected:
  std::shared_ptr<rafko_mainframe::RafkoSettings> m_settings;
  std::unique_ptr<google::protobuf::Arena> m_arena;

private:
  /**
   * @brief     Constructs an arena in case the provided settings doesn't contain any
   *
   * @param     settings    The @RafkoSettings instance to check for an existing arena implementation
   *
   * @return    The pointer to the arena should the @RafkoSettings instance not contain it.
   */
  static std::unique_ptr<google::protobuf::Arena> initialize_arena(rafko_mainframe::RafkoSettings& settings){
    if(nullptr == settings.get_arena_ptr())
      return std::make_unique<google::protobuf::Arena>();
      else return {};
  }
};

} /* namespace rafko_mainframe */

#endif /* RAFKO_AUTONOMOUS_ENTITY_H */

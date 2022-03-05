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

#ifndef RAFKO_ASSERTION_LOGGER_H
#define RAFKO_ASSERTION_LOGGER_H

#include "rafko_global.h"

#include <assert.h>
#include <memory>
#include <string>

namespace rafko_mainframe{

#ifndef NDEBUG
#define RFASSERT(condition) rafko_mainframe::RafkoAssertionLogger::rafko_assert(condition, __LINE__)
#else
#define RFASSERT(condition) ((void)condition)
#endif/* NDEBUG */

/**
 * @brief      Logger utility to create help identify problems in debug configurations, while
 *             not straining performance in release configurations
 */
class RafkoAssertionLogger{
public:
  std::unique_ptr<int> set_scope(){
    std::unique_ptr<int> next_scope = std::make_unique<int>();
    *next_scope = rand();
    if(!current_scope.expired()){
      //TODO: Log into the previous scope that another scope took over, and flush it
    }
    return next_scope;
  }

  constexpr static void rafko_assert(bool condition, std::size_t line_number){
    //TODO: If assertion fails, store a log message
    assert(condition);
  }

private:
  std::weak_ptr<int> current_scope;
};

} /* namespace rafko_mainframe */

#endif /* RAFKO_ASSERTION_LOGGER_H */

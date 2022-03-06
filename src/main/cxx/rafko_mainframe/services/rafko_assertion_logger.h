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
#include <sstream>
#include <mutex>
#include <chrono>

#include "spdlog/spdlog.h"

namespace rafko_mainframe{

#ifndef NDEBUG
#define RFASSERT(condition) rafko_mainframe::RafkoAssertionLogger::rafko_assert(condition)
#define RFASSERT_SCOPE(name) auto rafko_scope = rafko_mainframe::RafkoAssertionLogger::set_scope(#name);
#define RFASSERT_LOG(...) rafko_mainframe::RafkoAssertionLogger::rafko_log(__VA_ARGS__);
#define RFASSERT_LOGV(vec, ...) rafko_mainframe::RafkoAssertionLogger::rafko_log_vector(vec, __VA_ARGS__);
#else
#define RFASSERT(condition)
#define RFASSERT_SCOPE(name)
#define RFASSERT_LOG(...)
#define RFASSERT_LOGV(vec, ...)
#endif/* NDEBUG */

/**
 * @brief      Logger utility to create help identify problems in debug configurations, while
 *             not straining performance in release configurations
 */
class RafkoAssertionLogger{
public:
  static constexpr const std::string_view logs_folder = "logs";
  static std::shared_ptr<spdlog::logger> set_scope(std::string name);

  template<typename... Args>
  static void rafko_log(spdlog::format_string_t<Args...> fmt, Args &&... args){
    if(auto scope = current_scope.lock()){
      scope->log(spdlog::level::debug, fmt, args...);
    }
  }

  template<typename T, typename... Args>
  static void rafko_log_vector(std::vector<T> vec, spdlog::format_string_t<Args...> fmt, Args &&... args){
    if(auto scope = current_scope.lock()){
      scope->log(spdlog::level::debug, fmt, args...);
      std::stringstream vector_string;
      for(const T& e : vec){
        vector_string << "[" << e << "]";
      }
      scope->log(spdlog::level::debug, vector_string.str());
    }
  }

  static std::string get_current_scope_name(){
    return current_scope_name;
  }

  static void rafko_assert(bool condition);
private:
  static std::weak_ptr<spdlog::logger> current_scope;
  static std::string current_scope_name;
  static std::mutex scope_mutex;
  static bool keep_log;
};

} /* namespace rafko_mainframe */

#endif /* RAFKO_ASSERTION_LOGGER_H */

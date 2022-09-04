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

#include "rafko_mainframe/services/rafko_assertion_logger.hpp"

#if(RAFKO_USES_ASSERTLOGS)

#include <filesystem>
#include <chrono>
#include <date/date.h>

#include "spdlog/sinks/basic_file_sink.hpp"

namespace rafko_mainframe{

std::weak_ptr<spdlog::logger> RafkoAssertionLogger::m_currentScope;
std::string RafkoAssertionLogger::m_currentScopeName;
std::mutex RafkoAssertionLogger::m_scopeMutex;
bool RafkoAssertionLogger::m_keepLog = false;

std::shared_ptr<spdlog::logger> RafkoAssertionLogger::set_scope(std::string name){
  auto today = date::year_month_day{date::floor<date::days>(std::chrono::system_clock::now())};
  auto current_time = date::make_time(
    std::chrono::system_clock::now() - date::floor<date::days>(std::chrono::system_clock::now())
  );
  std::string scope_name = (
    name + "_" + (std::stringstream() << today).str() + "_" + (std::stringstream() << current_time).str()
  );
  spdlog::file_event_handlers handlers;
  handlers.after_close = [](spdlog::filename_t filename) {
    if(!m_keepLog) std::filesystem::remove(filename);
  };

  auto logger = spdlog::basic_logger_mt( scope_name, std::string(logs_folder) + "/" + scope_name + ".log", false, handlers);
  logger->set_pattern("[%H:%M:%S|%u][%^%L%$][thread %t] %v");
  logger->set_level(spdlog::level::trace);
  logger->flush_on(spdlog::level::err);
  /*!Note: no need to call spdlog::register_logger(logger);, because access is only through the pointer anyway */

  if(auto scope = m_currentScope.lock()){
    scope->info("Scope snatched by " + scope_name + "..");
    spdlog::drop(m_currentScopeName);
  }

  {
    std::lock_guard<std::mutex> my_lock(m_scopeMutex);
    m_currentScopeName = scope_name;
    m_currentScope = logger;
  }

  return logger;
}

void RafkoAssertionLogger::rafko_assert(bool condition, std::string file_name, std::uint32_t line_number){
  if(!condition){
    m_keepLog = true;
    if(auto scope = m_currentScope.lock()){ /* no need to use mutex here, since the underlying logger is thread-safe */
      scope->error("Assertion failure in file {}; line {}!", file_name, line_number);
      scope->flush();
    }
    spdlog::shutdown();
  }
  assert(condition);
}

} /* rafko_mainframe */

#endif/*(RAFKO_USES_ASSERTLOGS)*/

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

#include <catch2/catch_test_macros.hpp>

#include <fstream>

#include "rafko_mainframe/services/rafko_assertion_logger.hpp"

#include "test/test_utility.hpp"

namespace rafko_mainframe_test {

#if (RAFKO_USES_ASSERTLOGS)
TEST_CASE("Testing Logged Asserion System", "[assert]") {
  std::uint16_t small_value = 5;
  RFASSERT(4 <= 5);
  RFASSERT(5 == 5);
  RFASSERT(5 <= 6);
  RFASSERT(6u != small_value);
  RFASSERT(5u == small_value);

  std::string scope_name;
  {
    /* create new Scope */
    RFASSERT_SCOPE(test_scope);
    RFASSERT_LOG("Trying a message...");
    RFASSERT_LOGV(std::vector<int>(5), "This is a vector:");
    RFASSERT(true);
    scope_name =
        rafko_mainframe::RafkoAssertionLogger::get_current_scope_name();
  }
  spdlog::drop_all(); /* Need to drop the logger here, because otherwise async
                         handling of logfiles might cause false positive failure
                         here */
  std::string log_file_name =
      std::string(rafko_mainframe::RafkoAssertionLogger::logs_folder) + "/" +
      scope_name + ".log";
  bool file_exists;
  if (FILE *file = fopen(log_file_name.c_str(), "r")) {
    fclose(file);
    file_exists = true;
  } else {
    file_exists = false;
  }
  REQUIRE(!file_exists); /* No failed assertions were present so the logfile
                            should not exist */
}
#endif /*(RAFKO_USES_ASSERTLOGS)*/

} /* namespace rafko_mainframe_test */

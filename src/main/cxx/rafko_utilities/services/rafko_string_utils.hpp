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

#ifndef RAFKO_STRING_UTILS_H
#define RAFKO_STRING_UTILS_H

#include "rafko_global.hpp"

#include <regex>
#include <string>

namespace rafko_utilities {

/**
 * @brief      Replaces regex inside the source string with the substitution
 * until there are no more matches
 *
 * @param[in]  input_text         the text to replace the regex in
 * @param[in]  regex_to_replace   the regex to replace in the input string
 * @param[in]  substitute         the text to replace the regex matches to
 *
 * @return    The resulting string of the replaces
 */
std::string replace_all_in_string(std::string input_text,
                                  std::regex regex_to_replace,
                                  std::string substitute);

/**
 * @brief      Replaces regex inside the source string with the substitution
 * until there are no more matches
 *
 * @param[in]  characters_to_escape   a list of characters to escape
 * @param[in]  s                      the string to escape the characters in
 *
 * @return    The resulting string of the replaces
 */
std::string escape_string(std::string characters_to_escape, std::string s);

} /* namespace rafko_utilities */

#endif /* RAFKO_STRING_UTILS_H */

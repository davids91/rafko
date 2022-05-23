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

#include "rafko_utilities/services/rafko_string_utils.h"

namespace rafko_utilities {

std::string replace_all_in_string(std::string input_text, std::regex regex_to_replace, std::string substitute){
  std::string text = input_text;
  std::uint32_t matches_count = std::distance( /* https://stackoverflow.com/questions/8283735/count-number-of-matches */
    std::sregex_iterator(text.begin(), text.end(), regex_to_replace), std::sregex_iterator()
  );
  if(0u == matches_count){
    RFASSERT_LOG("Unneccesary replacement: couldn't find regex in '{}' initially!", text);
  }
  while(0u < matches_count){
    text = std::regex_replace(text, regex_to_replace, substitute);
    matches_count = std::distance(
      std::sregex_iterator(text.begin(), text.end(), regex_to_replace), std::sregex_iterator()
    );
  }
  return text;
}


} /* namespace rafko_utilities */

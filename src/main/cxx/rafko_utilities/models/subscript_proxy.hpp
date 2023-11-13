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

#ifndef SUBSCRIPT_PROXY_H
#define SUBSCRIPT_PROXY_H

#include "rafko_global.hpp"

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rafko_utilities {

template <typename Proxee = std::vector<double>>
class RAFKO_EXPORT SubscriptProxy {
public:
  using SubscriptDictionary = std::unordered_map<std::size_t, std::size_t>;
  using AssociationVector = std::vector<std::uint32_t>;

  SubscriptProxy(Proxee &object, AssociationVector associations)
      : m_object(&object), m_dictionary(std::make_shared<SubscriptDictionary>(
                               convert(associations))) {}

  SubscriptProxy(Proxee &object,
                 std::shared_ptr<SubscriptDictionary> dictionary = {})
      : m_object(&object), m_dictionary(dictionary) {}

  void update(Proxee &new_object) { m_object = &new_object; }

  typename Proxee::value_type &operator[](std::size_t index) {
    if (!m_dictionary)
      return (*m_object)[index];
    auto found_index = m_dictionary->find(index);
    if (found_index != m_dictionary->end())
      return (*m_object)[found_index->second];
    else
      return (*m_object)[index];
  }

  typename Proxee::iterator begin() { return m_object->begin(); }

  typename Proxee::iterator end() { return m_object->end(); }

private:
  Proxee *m_object;
  std::shared_ptr<SubscriptDictionary> m_dictionary;

  static SubscriptDictionary convert(AssociationVector vec) {
    SubscriptDictionary result;
    for (std::uint32_t i = 0; i < vec.size(); ++i) {
      result.insert({i, vec[i]});
    }
    return result;
  }
};

} /* namespace rafko_utilities */

#endif /* SUBSCRIPT_PROXY_H */

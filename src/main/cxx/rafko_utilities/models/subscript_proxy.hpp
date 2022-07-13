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

namespace rafko_utilities{

using SubscriptDictionary = std::unordered_map<std::size_t, std::size_t>;
template <typename Proxee = std::vector<double>>
class RAFKO_FULL_EXPORT SubscriptProxy{
public:
  SubscriptProxy(Proxee& object_, std::shared_ptr<SubscriptDictionary> dictionary_ = {})
  : object(&object_)
  , dictionary(dictionary_)
  {
  }

  void update(Proxee& new_object) {
    object = &new_object;
  }

  typename Proxee::value_type& operator[](std::size_t index){
    if(!dictionary)return (*object)[index];
    auto found_index = dictionary->find(index);
    if(found_index != dictionary->end())
      return (*object)[found_index->second];
      else return (*object)[index];
  }

  typename Proxee::iterator begin(){
    return object->begin();
  }

  typename Proxee::iterator end(){
    return object->end();
  }

private:
  Proxee* object;
  std::shared_ptr<SubscriptDictionary> dictionary;
};

} /* namespace rafko_utilities */

#endif/* SUBSCRIPT_PROXY_H */

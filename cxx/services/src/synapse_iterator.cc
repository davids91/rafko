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

#include "services/synapse_iterator.h"

namespace sparse_net_library{

int Synapse_iterator::operator[](int index) const{
  int result_index;
  int iteration_helper = 0;

  iterate_terminatable([&](int synapse_index){
    if(iteration_helper < index){
      ++iteration_helper;
      return true;
    }else{
      result_index = synapse_index;
      return false;
    }
  });
  if(iteration_helper != index)
    throw "Index Out of bounds with Synapse Iterator!";
  return result_index;
}

void Synapse_iterator::skim_unsafe(
  const RepeatedPtrField<Synapse_interval>& arg_synapse_interval,
  std::function< void(int, unsigned int) > do_for_each_synapse,
  uint32 interval_start, uint32 interval_size
){
  if((0 == interval_size)&&(arg_synapse_interval.size() > static_cast<int>(interval_start)))
    interval_size = arg_synapse_interval.size() - interval_start;
  for(uint32 synapse_iterator = interval_start; synapse_iterator < (interval_start + interval_size); ++synapse_iterator){
    do_for_each_synapse(arg_synapse_interval[synapse_iterator].starts(),arg_synapse_interval[synapse_iterator].interval_size());
  }
}

void Synapse_iterator::iterate_unsafe(
  const RepeatedPtrField<Synapse_interval>& arg_synapse_interval,
  std::function< void(int) > do_for_each_index,
  uint32 interval_start, uint32 interval_size
){
  if((0 == interval_size)&&(arg_synapse_interval.size() > static_cast<int>(interval_start)))
    interval_size = arg_synapse_interval.size() - interval_start;
  for(uint32 synapse_iterator = interval_start; synapse_iterator < (interval_start + interval_size); ++synapse_iterator){
    if(!is_index_input(arg_synapse_interval[synapse_iterator].starts())){
      for(uint32 input_iterator = 0; input_iterator < arg_synapse_interval[synapse_iterator].interval_size();++input_iterator){
        do_for_each_index(arg_synapse_interval[synapse_iterator].starts() + input_iterator);
      } /* For Every input inside a synapse */
    }else{ /* current @starts. element is from the input, iterate in a negative way */
      for(uint32 input_iterator = 0; input_iterator < arg_synapse_interval[synapse_iterator].interval_size();++input_iterator){
        do_for_each_index(arg_synapse_interval[synapse_iterator].starts() - input_iterator);
      } /* For Every input inside a synapse */
    }
  } /* For every synapse */
}

void Synapse_iterator::iterate_unsafe(
  const RepeatedPtrField<Synapse_interval>& arg_synapse_interval,
  std::function< void(unsigned int) > do_for_each_synapse,
  std::function< void(int) > do_for_each_index,
  uint32 interval_start, uint32 interval_size
){
  if((0 == interval_size)&&(arg_synapse_interval.size() > static_cast<int>(interval_start)))
    interval_size = arg_synapse_interval.size() - interval_start;
  for(uint32 synapse_iterator = interval_start; synapse_iterator < (interval_start + interval_size); ++synapse_iterator){
    do_for_each_synapse(arg_synapse_interval[synapse_iterator].interval_size());
    if(!is_index_input(arg_synapse_interval[synapse_iterator].starts())){
      for(uint32 input_iterator = 0; input_iterator < arg_synapse_interval[synapse_iterator].interval_size();++input_iterator){
        do_for_each_index(arg_synapse_interval[synapse_iterator].starts() + input_iterator);
      } /* For Every input inside a synapse */
    }else{ /* current @starts. element is from the input, iterate in a negative way */
      for(uint32 input_iterator = 0; input_iterator < arg_synapse_interval[synapse_iterator].interval_size();++input_iterator){
        do_for_each_index(arg_synapse_interval[synapse_iterator].starts() - input_iterator);
      } /* For Every input inside a synapse */
    }
  } /* For every synapse */
}

void Synapse_iterator::iterate_unsafe_terminatable(
  const RepeatedPtrField<Synapse_interval>& arg_synapse_interval,
  std::function< bool(int) > do_for_each_index,
  uint32 interval_start, uint32 interval_size
){
  if((0 == interval_size)&&(arg_synapse_interval.size() > static_cast<int>(interval_start)))
    interval_size = arg_synapse_interval.size() - interval_start;
  for(uint32 synapse_iterator = interval_start; synapse_iterator < (interval_start + interval_size); ++synapse_iterator){
    if(!is_index_input(arg_synapse_interval[synapse_iterator].starts())){
      for(uint32 input_iterator = 0; input_iterator < arg_synapse_interval[synapse_iterator].interval_size();++input_iterator){
        if(!do_for_each_index(arg_synapse_interval[synapse_iterator].starts() + input_iterator)){
          return;
        }
      } /* For Every input inside a synapse */
    }else{ /* current @starts. element is from the input, iterate in a negative way */
      for(uint32 input_iterator = 0; input_iterator < arg_synapse_interval[synapse_iterator].interval_size();++input_iterator){
        if(!do_for_each_index(arg_synapse_interval[synapse_iterator].starts() - input_iterator)){
          return;
        }
      } /* For Every input inside a synapse */
    }
  } /* For every synapse */
}

void Synapse_iterator::iterate_unsafe_terminatable(
  const RepeatedPtrField<Synapse_interval>& arg_synapse_interval,
  std::function< bool(unsigned int) > do_for_each_synapse,
  std::function< bool(int) > do_for_each_index,
  uint32 interval_start, uint32 interval_size
){
  if((0 == interval_size)&&(arg_synapse_interval.size() > static_cast<int>(interval_start)))
    interval_size = arg_synapse_interval.size() - interval_start;
  for(uint32 synapse_iterator = interval_start; synapse_iterator < (interval_start + interval_size); ++synapse_iterator){
    if(!do_for_each_synapse(arg_synapse_interval[synapse_iterator].interval_size())){
      return;
    }
    if(!is_index_input(arg_synapse_interval[synapse_iterator].starts())){      
      for(uint32 input_iterator = 0; input_iterator < arg_synapse_interval[synapse_iterator].interval_size();++input_iterator){
        if(!do_for_each_index(arg_synapse_interval[synapse_iterator].starts() + input_iterator)){
          return;
        }
      } /* For Every input inside a synapse */
    }else{ /* current @starts. element is from the input, iterate in a negative way */
      for(uint32 input_iterator = 0; input_iterator < arg_synapse_interval[synapse_iterator].interval_size();++input_iterator){
        if(!do_for_each_index(arg_synapse_interval[synapse_iterator].starts() - input_iterator)){
          return;
        }
      } /* For Every input inside a synapse */
    }
  } /* For every synapse */
}


} /* namespace sparse_net_library */

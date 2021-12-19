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

#ifndef SYNAPSE_ITERATOR_H
#define SYNAPSE_ITERATOR_H

#include "rafko_global.h"

#include <functional>
#include <stdexcept>

#include <google/protobuf/repeated_field.h>

#include "rafko_protocol/rafko_net.pb.h"

namespace rafko_net{

/**
 * @brief      This class describes a synapse iterator. Based on the given references
 *             it provides a hook to go through every index described by them.
 *             Since synapse iteration is a repeating occurence in Neuron routing,
 *             partial detail solution and in Solution input collection as well, this class
 *             aims to provide a standardized solution for it.
 *             @param[in]  do_for_each_synapse  A lambda function to process the synapses in each start of the synapse iteration.
 *                                Lambda parameter is the Synapse size
 *                                It shall return true if the iteration can continue.
 *
 *             @param[in]  do_for_each_index  A lambda function to process the indexes in each step of the synapse iteration.
 *                                Lambda parameter is the current Synapse index
 *                                It shall return true if the iteration can continue.
 *
 *             @param[in] interval_start  Defines the range the iteration shall visit
 *
 *             @param[in] interval_start  Defines the range the iteration shall visit
 *
 *             Default range is the whole of the synapses and the @do_for_each_synapse lambda is optional.
 *             Example: To iterate through a synapse set, a lambda for each synapse start, and for each element in that synapse:
 *             syn_iter.iterate([&](int synapse_start){},[&](int index){});
 */
template<typename Interval_type = IndexSynapseInterval>
class SynapseIterator{
public:
   SynapseIterator(const google::protobuf::RepeatedPtrField<Interval_type>& arg_synapse_interval)
  :  synapse_interval(arg_synapse_interval)
  ,  last_reached_synapse(0)
  ,  last_reached_index(0)
  { };

  void iterate(std::function< void(sint32) > do_for_each_index, uint32 interval_start = 0, uint32 interval_size_ = 0) const{
    iterate(synapse_interval, do_for_each_index, interval_start, interval_size_);
  }
  void iterate(std::function< void(Interval_type) > do_for_each_synapse, std::function< void(sint32) > do_for_each_index, uint32 interval_start = 0, uint32 interval_size_ = 0) const{
    iterate(synapse_interval, do_for_each_synapse, do_for_each_index, interval_start, interval_size_);
  }
  void iterate_terminatable(std::function< bool(sint32) > do_for_each_index, uint32 interval_start = 0, uint32 interval_size_ = 0) const{
    iterate_terminatable(synapse_interval, do_for_each_index, interval_start, interval_size_);
  }
  void iterate_terminatable(std::function< bool(Interval_type) > do_for_each_synapse, std::function< bool(sint32) > do_for_each_index, uint32 interval_start = 0, uint32 interval_size_ = 0) const{
    iterate_terminatable(synapse_interval, do_for_each_synapse, do_for_each_index, interval_start, interval_size_);
  }
  void skim(std::function< void(Interval_type) > do_for_each_synapse, uint32 interval_start = 0, uint32 interval_size_ = 0) const{
    skim(synapse_interval, do_for_each_synapse, interval_start, interval_size_);
  }
  void skim_terminatable(std::function< bool(Interval_type) > do_for_each_synapse, uint32 interval_start = 0, uint32 interval_size_ = 0) const{
    skim_terminatable(synapse_interval, do_for_each_synapse, interval_start, interval_size_);
  }

  static void skim(
    const google::protobuf::RepeatedPtrField<Interval_type>& arg_synapse_interval,
    std::function< void(Interval_type) > do_for_each_synapse,
    uint32 interval_start = 0, uint32 interval_size_ = 0
  ){
    uint32 interval_size = get_number_of_synapses_to_iterate(arg_synapse_interval, interval_start, interval_size_);
    for(uint32 syn_iter = interval_start; syn_iter < (interval_start + interval_size); ++syn_iter){
      do_for_each_synapse(arg_synapse_interval[syn_iter]);
    }
  }

  static void iterate(
    const google::protobuf::RepeatedPtrField<Interval_type>& arg_synapse_interval,
    std::function< void(sint32) > do_for_each_index,
    uint32 interval_start = 0, uint32 interval_size_ = 0
  ){
    uint32 interval_size = get_number_of_synapses_to_iterate(arg_synapse_interval, interval_start, interval_size_);
    for(uint32 syn_iter = interval_start; syn_iter < (interval_start + interval_size); ++syn_iter){
      uint32 this_interval_size = arg_synapse_interval[syn_iter].interval_size();
      uint32 this_interval_start = arg_synapse_interval[syn_iter].starts();
      if(!is_index_input(this_interval_start)){
        for(uint32 input_iterator = 0; input_iterator < this_interval_size ;++input_iterator){
          do_for_each_index(this_interval_start + input_iterator);
        }
      }
      else /* current @starts. element is from the input, iterate in a negative way */
        for(uint32 input_iterator = 0; input_iterator < this_interval_size;++input_iterator){
          do_for_each_index(this_interval_start - input_iterator);
        }
    } /* For every synapse */
  }

  static void iterate(
    const google::protobuf::RepeatedPtrField<Interval_type>& arg_synapse_interval,
    std::function< void(Interval_type) > do_for_each_synapse,
    std::function< void(sint32) > do_for_each_index,
    uint32 interval_start = 0, uint32 interval_size_ = 0
  ){
    uint32 interval_size = get_number_of_synapses_to_iterate(arg_synapse_interval, interval_start, interval_size_);
    for(uint32 syn_iter = interval_start; syn_iter < (interval_start + interval_size); ++syn_iter){
      do_for_each_synapse(arg_synapse_interval[syn_iter]);
      uint32 this_interval_size = arg_synapse_interval[syn_iter].interval_size();
      uint32 this_interval_start = arg_synapse_interval[syn_iter].starts();
      if(!is_index_input(this_interval_start)){
        for(uint32 input_iterator = 0; input_iterator < this_interval_size;++input_iterator){
          do_for_each_index(this_interval_start + input_iterator);
        }
      }else{ /* current @starts. element is from the input, iterate in a negative way */
        for(sint32 input_iterator = 0; static_cast<uint32>(input_iterator) < this_interval_size;++input_iterator){
          do_for_each_index(this_interval_start - input_iterator);
        }
      }
    } /* For every synapse */
  }

  static void skim_terminatable(
    const google::protobuf::RepeatedPtrField<Interval_type>& arg_synapse_interval,
    std::function< bool(Interval_type) > do_for_each_synapse,
    uint32 interval_start = 0, uint32 interval_size_ = 0
  ){
    uint32 interval_size = get_number_of_synapses_to_iterate(arg_synapse_interval, interval_start, interval_size_);
    for(uint32 syn_iter = interval_start; syn_iter < (interval_start + interval_size); ++syn_iter)
      if(!do_for_each_synapse(arg_synapse_interval[syn_iter]))
        return;
  }

  static void iterate_terminatable(
    const google::protobuf::RepeatedPtrField<Interval_type>& arg_synapse_interval,
    std::function< bool(sint32) > do_for_each_index,
    uint32 interval_start = 0, uint32 interval_size_ = 0
  ){
    uint32 interval_size = get_number_of_synapses_to_iterate(arg_synapse_interval, interval_start, interval_size_);
    for(uint32 syn_iter = interval_start; syn_iter < (interval_start + interval_size); ++syn_iter){
      uint32 this_interval_size = arg_synapse_interval[syn_iter].interval_size();
      uint32 this_interval_start = arg_synapse_interval[syn_iter].starts();
      if(!is_index_input(this_interval_start)){
        for(uint32 input_iterator = 0; input_iterator < this_interval_size;++input_iterator)
          if(!do_for_each_index(this_interval_start + input_iterator))
            return;
      }else{ /* current @starts. element is from the input, iterate in a negative way */
        for(uint32 input_iterator = 0; input_iterator < this_interval_size;++input_iterator)
          if(!do_for_each_index(this_interval_start - input_iterator))
            return;
      }
    } /* For every synapse */
  }

  static void iterate_terminatable(
    const google::protobuf::RepeatedPtrField<Interval_type>& arg_synapse_interval,
    std::function< bool(Interval_type) > do_for_each_synapse,
    std::function< bool(sint32) > do_for_each_index,
    uint32 interval_start = 0, uint32 interval_size_ = 0
  ){
    uint32 interval_size = get_number_of_synapses_to_iterate(arg_synapse_interval, interval_start, interval_size_);
    for(uint32 syn_iter = interval_start; syn_iter < (interval_start + interval_size); ++syn_iter){
      if(!do_for_each_synapse(arg_synapse_interval[syn_iter]))
        return;
      uint32 this_interval_size = arg_synapse_interval[syn_iter].interval_size();
      uint32 this_interval_start = arg_synapse_interval[syn_iter].starts();
      if(!is_index_input(this_interval_start)){
        for(uint32 input_iterator = 0; input_iterator < this_interval_size;++input_iterator)
          if(!do_for_each_index(this_interval_start + input_iterator))
            return;
      }else{ /* current @starts. element is from the input, iterate in a negative way */
        for(uint32 input_iterator = 0; input_iterator < this_interval_size;++input_iterator)
          if(!do_for_each_index(this_interval_start - input_iterator))
            return;
      }
    } /* For every synapse */
  }

  /**
   * @brief      Direct access to an indvidual synapse index. Warning! very greedy!
   *             Instead of overflow it returns with 0 in case the given index is bigger, than the synapse size
   *
   * @param[in]  index  The index
   *
   * @return     The Synapse index under the @index-th step into the iteration
   */
  int operator[](int index){
    if(0 == size())throw std::runtime_error("Empty synapse iterator reached for subscript!");
    sint32 result_index;
    uint32 previous_last_reached_index = 0;
    sint32 iteration_helper = 0;
    uint32 synapse_start = 0;

    if(static_cast<sint32>(last_reached_index) <= index){
      synapse_start = last_reached_synapse;
      iteration_helper = last_reached_index;
    }else last_reached_synapse = 0;

    iterate_terminatable([&](Interval_type interval_synapse){
      parameter_not_used(interval_synapse);
      ++last_reached_synapse;
      last_reached_index = iteration_helper;
      previous_last_reached_index = last_reached_index;
      return true;
    },[&](sint32 synapse_index){
      if(iteration_helper < index){
        ++iteration_helper;
        return true;
      }else{
        result_index = synapse_index;
        return false;
      }
    },synapse_start);
    if(iteration_helper != index)
      throw std::runtime_error("Index Out of bounds with Synapse Iterator!");
    --last_reached_synapse;
    last_reached_index = previous_last_reached_index;
    return result_index;
  }

  /**
   * @brief      Gives back a copy of the synapse under the given index:
   *             In a synapse with multiple intervals, the index refers to
   *             the number of contained indices, not the number of intervals.
   *
   * @param[in]  index  The index
   *
   * @return     The interval synapse
   */
  Interval_type synapse_under(sint32 index){
    if(0 == size())throw "Empty synapse iterator reached for query!";
    sint32 iteration_helper = 0;
    Interval_type result;

    iterate_terminatable([&](Interval_type interval_synapse){
      result = interval_synapse;
      return true;
    },[&](sint32 synapse_index){
      parameter_not_used(synapse_index);
      if(iteration_helper < index){
        ++iteration_helper;
        return true; /* Found the synapse we have been looking for */
      }else return false; /* Continue searching.. */
    });
    if(iteration_helper != index)
      throw "Index Out of bounds with Synapse Iterator!";
    return result;
  }

  /**
   * @brief      Returns the overall number of inputs
   *
   * @return     Returns the overall number of inputs
   */
  uint32 size() const{
    uint32 number_of_inputs = 0;
    skim([&](Interval_type interval){
      number_of_inputs += interval.interval_size();
    });
    return number_of_inputs;
  }

  /**
   * @brief      Return the number of input synapses in the iterator.
   *
   * @return     Size of the Repeatedfiled for the  synapse_intervals
   */
  uint32 number_of_synapses() const{
    return synapse_interval.size();
  }

  /**
   * @brief      Give back the last element of the synapse
   *
   * @return     the last index of the synapse
   */
  sint32 back() const{
    if(0 < synapse_interval.size()){
      sint32 last_index = synapse_interval[synapse_interval.size()-1].starts();
      if(is_index_input(last_index)) last_index -= synapse_interval[synapse_interval.size()-1].interval_size() - 1;
        else last_index += synapse_interval[synapse_interval.size()-1].interval_size() - 1;
      return last_index;
    }else throw std::runtime_error("Last index requested from empty synapse!");
  }

  /**
   * @brief      Add back the last stored synapse interval
   *
   * @return     Synapse interval defined by template function
   */
  Interval_type last_synapse() const{
    if(0 < synapse_interval.size()){
      return synapse_interval[synapse_interval.size()-1];
    }else throw std::runtime_error("Last item requested from empty synapse!");
  }

  /**
   * @brief      Determines whether the specified index is index taken from the inputs, rather than internally.
   *
   * @param[in]  index  The index
   *
   * @return     True if the specified index is index input, False otherwise.
   */
  static bool is_index_input(sint32 index){
    return(0 > index);
  }


  /**
   * @brief      Determines whether the specified input interval is pointing to the inputs, rather than internal Neuron data.
   *
   * @param[in]  interval   A const reference to the interval to examine
   *
   * @return     True if the specified interval is pointing to network input, False otherwise.
   */
  static bool is_synapse_input(const Interval_type& interval){
    return(is_index_input(interval.starts()));
  }

  /**
   * @brief      Converts synapse input index to an index usable in an input array
   *
   * @param[in]  index  Synapse index
   *
   * @return     Index in the input array based on the synapse input index
   */
  static sint32 synapse_index_from_input_index(uint32 index){
    return (static_cast<sint32>(index) * (-1) - 1);
  }

  /**
   * @brief      Converts array indexes of an array to Synapse input indexes to be stored in messages
   *
   * @param[in]  index  The input array index
   *
   * @return     Synapse index
   */
  static uint32 input_index_from_synapse_index(sint32 index){
    if(0 > index) return (static_cast<uint32>(index) * (-1) - 1);
      else throw std::runtime_error("Synapse index is not negative, as it should be, when queried for input index! ");
  }

private:
  const google::protobuf::RepeatedPtrField<Interval_type>& synapse_interval;
  uint32 last_reached_synapse;
  uint32 last_reached_index;
  static uint32 interval_size; /* temporary variable */

  /**
   * @brief      Gets the number of synapses to iterate based on the provided start and size values.
   *             Checks wether the arguments related validity, and returns the number of synapses to iterate over.
   *             It sets the mutable helper @interval_size to the correct iteration value
   *
   * @param      arg_synapse_interval  The argument synapse interval
   * @param[in]  interval_start        The interval start
   * @param[in]  interval_size_        The interval size
   *
   */
  static uint32 get_number_of_synapses_to_iterate(const google::protobuf::RepeatedPtrField<Interval_type>& arg_synapse_interval, uint32 interval_start, uint32 interval_size_){
    if(0 == arg_synapse_interval.size())
      return 0;
    else if( (0 == interval_size_)&&(arg_synapse_interval.size() > static_cast<sint32>(interval_start)) )
      return (arg_synapse_interval.size() - interval_start);
    else if(0 == interval_size_)
      throw std::runtime_error("Incorrect synapse range start!");
    else return interval_size_;
  }
};

} /* namespace rafko_net */

#endif /* SYNAPSE_ITERATOR_H */

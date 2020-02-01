#ifndef SYNAPSE_ITERATOR_H
#define SYNAPSE_ITERATOR_H

#include "sparse_net_global.h"

#include <functional>
#include <google/protobuf/repeated_field.h>

#include "gen/common.pb.h"

namespace sparse_net_library{

using std::reference_wrapper;
using google::protobuf::RepeatedPtrField;

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
 *             synapse_iterator.iterate([&](int synapse_start){},[&](int index){});
 */
class Synapse_iterator{
public:
   Synapse_iterator(const RepeatedPtrField<Synapse_interval>& arg_synapse_interval)
  : synapse_interval(arg_synapse_interval)
  { };

  void iterate(std::function< void(int) > do_for_each_index, uint32 interval_start, uint32 interval_size = 0) const{
    if((0 == interval_size)&&(synapse_interval.get().size() > static_cast<int>(interval_start)))
      interval_size = synapse_interval.get().size() - interval_start;
    else if(0 == interval_size) throw "Incorrect synapse range start!";
    if(static_cast<int>(interval_start + interval_size) <= synapse_interval.get().size()){ 
      iterate_unsafe(do_for_each_index,interval_start,interval_size);
    }else throw "Incorrect Synapse range!";
  }
  void iterate(std::function< void(unsigned int) > do_for_each_synapse, std::function< void(int) > do_for_each_index, uint32 interval_start, uint32 interval_size = 0) const{
    if((0 == interval_size)&&(synapse_interval.get().size() > static_cast<int>(interval_start)))
      interval_size = synapse_interval.get().size() - interval_start;
    else if(0 == interval_size) throw "Incorrect synapse range start!";
    if(static_cast<int>(interval_start + interval_size) <= synapse_interval.get().size()){ 
      iterate_unsafe(do_for_each_synapse,do_for_each_index,interval_start,interval_size);
    }else throw "Incorrect Synapse range!";
  }
  void iterate_terminatable(std::function< bool(int) > do_for_each_index, uint32 interval_start, uint32 interval_size = 0) const{
    if((0 == interval_size)&&(synapse_interval.get().size() > static_cast<int>(interval_start)))
      interval_size = synapse_interval.get().size() - interval_start;
    else if(0 == interval_size) throw "Incorrect synapse range start!";
    if(static_cast<int>(interval_start + interval_size) <= synapse_interval.get().size()){ 
      iterate_unsafe_terminatable(do_for_each_index,interval_start,interval_size);
    }else throw "Incorrect Synapse range!";
  }
  void iterate_terminatable(std::function< bool(unsigned int) > do_for_each_synapse, std::function< bool(int) > do_for_each_index, uint32 interval_start, uint32 interval_size = 0) const{
    if((0 == interval_size)&&(synapse_interval.get().size() > static_cast<int>(interval_start)))
      interval_size = synapse_interval.get().size() - interval_start;
    else if(0 == interval_size) throw "Incorrect synapse range start!";
    if(static_cast<int>(interval_start + interval_size) <= synapse_interval.get().size()){ 
      iterate_unsafe_terminatable(do_for_each_synapse,do_for_each_index,interval_start,interval_size);
    }else throw "Incorrect Synapse range!";
  }
  void skim(std::function< void(int, unsigned int) > do_for_each_synapse) const{
    skim_unsafe(do_for_each_synapse,0,synapse_interval.get().size());
  }
  void iterate(std::function< void(int) > do_for_each_index) const{
    iterate_unsafe(do_for_each_index, 0, synapse_interval.get().size());
  }
  void iterate(std::function< void(unsigned int) > do_for_each_synapse, std::function< void(int) > do_for_each_index) const{
    iterate_unsafe(do_for_each_synapse, do_for_each_index, 0, synapse_interval.get().size());
  }
  void iterate_terminatable(std::function< bool(int) > do_for_each_index) const{
    iterate_unsafe_terminatable(do_for_each_index, 0, synapse_interval.get().size());
  }
  void iterate_terminatable(std::function< bool(unsigned int) > do_for_each_synapse, std::function< bool(int) > do_for_each_index) const{
    iterate_unsafe_terminatable(do_for_each_synapse,do_for_each_index, 0, synapse_interval.get().size());
  }

  void skim_unsafe(std::function< void(int, unsigned int) > do_for_each_synapse, uint32 interval_start, uint32 interval_size = 0) const;
  void iterate_unsafe(std::function< void(int) > do_for_each_index, uint32 interval_start, uint32 interval_size = 0) const;
  void iterate_unsafe(std::function< void(unsigned int) > do_for_each_synapse, std::function< void(int) > do_for_each_index, uint32 interval_start, uint32 interval_size = 0) const;
  void iterate_unsafe_terminatable(std::function< bool(int) > do_for_each_index, uint32 interval_start, uint32 interval_size = 0) const;
  void iterate_unsafe_terminatable(std::function< bool(unsigned int) > do_for_each_synapse, std::function< bool(int) > do_for_each_index, uint32 interval_start, uint32 interval_size = 0) const;

  /**
   * @brief      Direct access to an indvidual synapse index. Warning! very greedy!
   *             Instead of overflow it returns with 0 in case the given index is bigger, than the synapse size
   *
   * @param[in]  index  The index
   *
   * @return     The Synapse index under the @index-th step into the iteration
   */
  int operator[](int index) const;

  /**
   * @brief      Returns the overall number of inputs
   *
   * @return     Returns the overall number of inputs
   */
  uint32 size(void) const{
    uint32 number_of_inputs = 0;
    skim([&](int synapse_starts, unsigned int synapse_size){
      number_of_inputs += synapse_size;
    });
    return number_of_inputs;
  }

  /**
   * @brief      Give back the last element of the synapse
   *
   * @return     the last index of the synapse
   */
  int back(void) const{
    if(0 < synapse_interval.get().size()){
      int last_index = synapse_interval.get().Get(synapse_interval.get().size()-1).starts();
      if(is_index_input(last_index)) last_index -= synapse_interval.get().Get(synapse_interval.get().size()-1).interval_size() - 1;
        else last_index += synapse_interval.get().Get(synapse_interval.get().size()-1).interval_size() - 1;
      return last_index;
    }else throw "Last item requested from empty synapse!";
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
      else throw "Synapse index is not negative, as it should be, when queried for input index! ";
  }

private:
  /**
   * The index of the first element in every synapse
   */
  reference_wrapper<const RepeatedPtrField<Synapse_interval>> synapse_interval;
};

} /* namespace sparse_net_library */

#endif /* SYNAPSE_ITERATOR_H */

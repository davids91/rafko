#ifndef SYNAPSE_ITERATOR_H
#define SYNAPSE_ITERATOR_H

#include "sparse_net_global.h"

#include <functional>
#include <google/protobuf/repeated_field.h>

namespace sparse_net_library{

/**
 * @brief      This class describes a synapse iterator. Based on the given references
 *             it provides a hook to go through every index described by them.
 *             Since synapse iteration is a repeating occurence in Neuron routing, 
 *             partial detail solution and in Solution input collection as well, this class 
 *             aims to provide a standardized solution for it.
 */
class Synapse_iterator{
public:
  Synapse_iterator(const google::protobuf::RepeatedField<int>& arg_starts, const google::protobuf::RepeatedField<unsigned int>& arg_sizes);

  /**
   * @brief      Iterates through the synapse if the sizes of each synapse RepeatedFIled are equal.
   *
   * @param[in]  do_for_each_index  A lambda function to process the indexes in each step of the synapse iteration.
   *                                It shall return true if the iteration can continue.
   */
  void iterate(std::function< bool(int) > do_for_each_index);
  void iterate(std::function< bool(int) > do_for_each_index, uint32 interval_start, uint32 interval_size);

  /**
   * @brief      Iterates through the synapse irregardless if the sizes of each synapse RepeatedFIled are equal.
   *
   * @param[in]  do_for_each_index  A lambda function to process the indexes in each step of the synapse iteration.
   *                                It shall return true if the iteration can continue.
   */
  void iterate_unsafe(std::function< bool(int) > do_for_each_index);
  void iterate_unsafe(std::function< bool(int) > do_for_each_index, uint32 interval_start, uint32 interval_size);

  inline static bool is_index_input(sint32 index){
    return(0 > index);
  }

  inline static sint32 synapse_index_from_input_index(uint32 index){
    return (static_cast<sint32>(index) * (-1) - 1);
  }

  inline static uint32 input_index_from_synapse_index(sint32 index){
    if(0 > index) return (static_cast<uint32>(index) * (-1) - 1);
      else throw "Synapse index is not negative, as it should be, when queried for input index! ";
  }

private:
  /**
   * The index of the first element in every synapse
   */
  const google::protobuf::RepeatedField<int>& starts;

  /**
   * The number of elements in every synapse
   */
  const google::protobuf::RepeatedField<unsigned int>& sizes;
};

} /* namespace sparse_net_library */

#endif /* SYNAPSE_ITERATOR_H */

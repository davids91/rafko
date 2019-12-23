#include "services/synapse_iterator.h"

namespace sparse_net_library{

Synapse_iterator::Synapse_iterator(const google::protobuf::RepeatedField<int>& arg_starts, const google::protobuf::RepeatedField<unsigned int>& arg_sizes)
  : starts(arg_starts), sizes(arg_sizes){};

void Synapse_iterator::iterate(std::function< bool(int) > do_for_each_index){
  if(sizes.size() == starts.size()){
    iterate_unsafe(do_for_each_index);
  } else throw "Incorrect Synapse members sizes! "; /*!#7 */
}

void Synapse_iterator::iterate(std::function< bool(int) > do_for_each_index, uint32 interval_start, uint32 interval_size){
  if( /* Check if synapse component sizes are equal */
      (sizes.size() == starts.size())
      &&(static_cast<int>(interval_start + interval_size) <= starts.size()) 
    ){ /* and the given interval is valid */
    iterate_unsafe(do_for_each_index,interval_start,interval_size);
  }else throw "Incorrect Synapse range!";
}

void Synapse_iterator::iterate_unsafe(std::function< bool(int) > do_for_each_index){
  iterate_unsafe(do_for_each_index, 0, starts.size());
}

void Synapse_iterator::iterate_unsafe(std::function< bool(int) > do_for_each_index, uint32 interval_start, uint32 interval_size){
  for(uint32 synapse_iterator = interval_start; synapse_iterator < (interval_start + interval_size); ++synapse_iterator){
    if(!is_index_input(starts.Get(synapse_iterator))){
      for(uint32 input_iterator = 0; input_iterator < sizes.Get(synapse_iterator);++input_iterator){
        if(!do_for_each_index(starts.Get(synapse_iterator) + input_iterator)){
          return;
        }
      } /* For Every input inside a synapse */
    }else{ /* current @starts element is from the input, iterate in a negative way */
      for(uint32 input_iterator = 0; input_iterator < sizes.Get(synapse_iterator);++input_iterator){
        if(!do_for_each_index(starts.Get(synapse_iterator) - input_iterator)){
          return;
        }
      } /* For Every input inside a synapse */
    }
  } /* For every synapse */
}


} /* namespace sparse_net_library */

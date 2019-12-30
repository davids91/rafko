#include "services/synapse_iterator.h"

namespace sparse_net_library{

int Synapse_iterator::operator[](int index){
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

void Synapse_iterator::skim_unsafe(std::function< void(int, unsigned int) > do_for_each_synapse, uint32 interval_start, uint32 interval_size) const{
  if((0 == interval_size)&&(synapse_interval.get().size() > static_cast<int>(interval_start)))
    interval_size = synapse_interval.get().size() - interval_start;
  for(uint32 synapse_iterator = interval_start; synapse_iterator < (interval_start + interval_size); ++synapse_iterator){
    do_for_each_synapse(synapse_interval.get().Get(synapse_iterator).starts(),synapse_interval.get().Get(synapse_iterator).interval_size());
  }
}

void Synapse_iterator::iterate_unsafe(std::function< void(int) > do_for_each_index, uint32 interval_start, uint32 interval_size) const{
  if((0 == interval_size)&&(synapse_interval.get().size() > static_cast<int>(interval_start)))
    interval_size = synapse_interval.get().size() - interval_start;
  for(uint32 synapse_iterator = interval_start; synapse_iterator < (interval_start + interval_size); ++synapse_iterator){
    if(!is_index_input(synapse_interval.get().Get(synapse_iterator).starts())){
      for(uint32 input_iterator = 0; input_iterator < synapse_interval.get().Get(synapse_iterator).interval_size();++input_iterator){
        do_for_each_index(synapse_interval.get().Get(synapse_iterator).starts() + input_iterator);
      } /* For Every input inside a synapse */
    }else{ /* current @starts.get(). element is from the input, iterate in a negative way */
      for(uint32 input_iterator = 0; input_iterator < synapse_interval.get().Get(synapse_iterator).interval_size();++input_iterator){
        do_for_each_index(synapse_interval.get().Get(synapse_iterator).starts() - input_iterator);
      } /* For Every input inside a synapse */
    }
  } /* For every synapse */
}

void Synapse_iterator::iterate_unsafe(std::function< void(unsigned int) > do_for_each_synapse, std::function< void(int) > do_for_each_index, uint32 interval_start, uint32 interval_size) const{
  if((0 == interval_size)&&(synapse_interval.get().size() > static_cast<int>(interval_start)))
    interval_size = synapse_interval.get().size() - interval_start;
  for(uint32 synapse_iterator = interval_start; synapse_iterator < (interval_start + interval_size); ++synapse_iterator){
    do_for_each_synapse(synapse_interval.get().Get(synapse_iterator).interval_size());
    if(!is_index_input(synapse_interval.get().Get(synapse_iterator).starts())){
      for(uint32 input_iterator = 0; input_iterator < synapse_interval.get().Get(synapse_iterator).interval_size();++input_iterator){
        do_for_each_index(synapse_interval.get().Get(synapse_iterator).starts() + input_iterator);
      } /* For Every input inside a synapse */
    }else{ /* current @starts.get(). element is from the input, iterate in a negative way */
      for(uint32 input_iterator = 0; input_iterator < synapse_interval.get().Get(synapse_iterator).interval_size();++input_iterator){
        do_for_each_index(synapse_interval.get().Get(synapse_iterator).starts() - input_iterator);
      } /* For Every input inside a synapse */
    }
  } /* For every synapse */
}

void Synapse_iterator::iterate_unsafe_terminatable(std::function< bool(int) > do_for_each_index, uint32 interval_start, uint32 interval_size) const{
  if((0 == interval_size)&&(synapse_interval.get().size() > static_cast<int>(interval_start)))
    interval_size = synapse_interval.get().size() - interval_start;
  for(uint32 synapse_iterator = interval_start; synapse_iterator < (interval_start + interval_size); ++synapse_iterator){
    if(!is_index_input(synapse_interval.get().Get(synapse_iterator).starts())){
      for(uint32 input_iterator = 0; input_iterator < synapse_interval.get().Get(synapse_iterator).interval_size();++input_iterator){
        if(!do_for_each_index(synapse_interval.get().Get(synapse_iterator).starts() + input_iterator)){
          return;
        }
      } /* For Every input inside a synapse */
    }else{ /* current @starts.get(). element is from the input, iterate in a negative way */
      for(uint32 input_iterator = 0; input_iterator < synapse_interval.get().Get(synapse_iterator).interval_size();++input_iterator){
        if(!do_for_each_index(synapse_interval.get().Get(synapse_iterator).starts() - input_iterator)){
          return;
        }
      } /* For Every input inside a synapse */
    }
  } /* For every synapse */
}

void Synapse_iterator::iterate_unsafe_terminatable(std::function< bool(unsigned int) > do_for_each_synapse, std::function< bool(int) > do_for_each_index, uint32 interval_start, uint32 interval_size) const{
  if((0 == interval_size)&&(synapse_interval.get().size() > static_cast<int>(interval_start)))
    interval_size = synapse_interval.get().size() - interval_start;
  for(uint32 synapse_iterator = interval_start; synapse_iterator < (interval_start + interval_size); ++synapse_iterator){
    if(!do_for_each_synapse(synapse_interval.get().Get(synapse_iterator).interval_size())){
      return;
    }
    if(!is_index_input(synapse_interval.get().Get(synapse_iterator).starts())){      
      for(uint32 input_iterator = 0; input_iterator < synapse_interval.get().Get(synapse_iterator).interval_size();++input_iterator){
        if(!do_for_each_index(synapse_interval.get().Get(synapse_iterator).starts() + input_iterator)){
          return;
        }
      } /* For Every input inside a synapse */
    }else{ /* current @starts.get(). element is from the input, iterate in a negative way */
      for(uint32 input_iterator = 0; input_iterator < synapse_interval.get().Get(synapse_iterator).interval_size();++input_iterator){
        if(!do_for_each_index(synapse_interval.get().Get(synapse_iterator).starts() - input_iterator)){
          return;
        }
      } /* For Every input inside a synapse */
    }
  } /* For every synapse */
}


} /* namespace sparse_net_library */

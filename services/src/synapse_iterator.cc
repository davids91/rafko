#include "services/synapse_iterator.h"

namespace sparse_net_library{

Synapse_iterator::Synapse_iterator(google::protobuf::RepeatedField<uint32>& arg_sizes, google::protobuf::RepeatedField<sint32>& arg_starts)
  : sizes(arg_sizes), starts(arg_starts){};

void Synapse_iterator::run(std::function< void(int) >& do_for_each_index){
  if(sizes.size() == starts.size()){
    for(int synapse_iterator = 0; synapse_iterator < sizes.size(); ++synapse_iterator){
      for(uint32 input_iterator = 0; input_iterator < sizes.Get(synapse_iterator);++input_iterator){
        do_for_each_index(starts.Get(synapse_iterator) + input_iterator);
      } /* For Every input inside a synapse */
    } /* For every synapse */
  } else throw "Incorrect Synapse members sizes! "; /*!#7 */
}

} /* namespace sparse_net_library */

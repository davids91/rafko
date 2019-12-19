#ifndef SYNAPSE_ITERATOR_H
#define SYNAPSE_ITERATOR_H
#include "sparse_net_global.h"

#include <functional>
#include <google/protobuf/repeated_field.h>

namespace sparse_net_library{

class Synapse_iterator{
public:
  Synapse_iterator(google::protobuf::RepeatedField<uint32>& arg_sizes, google::protobuf::RepeatedField<sint32>& arg_starts);
  void run(std::function< void(int) >& do_for_each_index);

private:
  google::protobuf::RepeatedField<uint32>& sizes;
  google::protobuf::RepeatedField<sint32>& starts;
};

} /* namespace sparse_net_library */

#endif /* SYNAPSE_ITERATOR_H */

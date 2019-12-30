#ifndef PARTIAL_SOLUTION_BUILDER_H
#define PARTIAL_SOLUTION_BUILDER_H

#include "sparse_net_global.h"
#include "gen/sparse_net.pb.h"
#include "gen/solution.pb.h"
#include "services/synapse_iterator.h"

#include <functional>

namespace sparse_net_library{

using std::reference_wrapper;
using google::protobuf::RepeatedPtrField;

/**
 * @brief      Front-end to create partial solution objects by adding Neurons into them.
 */
class Partial_solution_builder{

public:
  Partial_solution_builder(const SparseNet& net, Partial_solution& partial_ref)
  : net(net), partial(partial_ref), input_synapse(partial_ref.input_data()){
    previous_neuron_input_source = neuron_input_none;
  };

  /**
   * @brief      Adds a neuron to partial solution.
   *
   * @param      net  The sparse net to read the neuron from
   * @param      neuron_index  the index of the neuron inside the net
   */
  void add_neuron_to_partial_solution(uint32 neuron_index);

  static void add_to_synapse(int index, uint32& current_synapse_count, RepeatedPtrField<Synapse_interval>* synapse_intervals){
    if((0 < synapse_intervals->size())&&(0 < current_synapse_count)){ /* Currently building a synapse already */
      ++current_synapse_count;
      synapse_intervals->Mutable(synapse_intervals->size()-1)->set_interval_size(current_synapse_count);
    }else{ /* Opening up a totally new Neuron Synapse */
      Synapse_interval new_interval;
      new_interval.set_starts(index);
      new_interval.set_interval_size(1);
      *synapse_intervals->Add() = new_interval;
      current_synapse_count = 1;
    }
  }

private:
  bool look_for_neuron_input(int neuron_input_index);
  bool look_for_neuron_input_internally(uint32 neuron_input_index);
  reference_wrapper<const SparseNet> net;
  reference_wrapper<Partial_solution> partial;
  Synapse_iterator input_synapse;
  uint32 neuron_synapse_count = 0;
  uint32 partial_input_synapse_count = 0;
  int previous_neuron_input_index;
  uint8 previous_neuron_input_source;
  static const uint8 neuron_input_none = 0;
  static const uint8 neuron_input_internal = 1;
  static const uint8 neuron_input_external = 2;
};

} /* namespace sparse_net_library */

#endif /* PARTIAL_SOLUTION_BUILDER_H */
#include <time.h>

#include <iostream>

#include "services/sparse_net_builder.h"

#include "models/dense_net_weight_initializer.h"
#include "models/neuron_info.h"
#include "services/synapse_iterator.h"

namespace sparse_net_library {

using std::shared_ptr;

SparseNetBuilder& SparseNetBuilder::input_size(uint32 size){
  arg_input_size = size;
  is_input_size_set = true;
  return *this;
}

SparseNetBuilder& SparseNetBuilder::output_neuron_number(uint32 size){
  arg_output_neuron_number = size;
  is_output_neuron_number_set = true;
  return *this;
}

SparseNetBuilder& SparseNetBuilder::expectedInputRange(sdouble32 range){
  arg_expected_input_range = range;
  is_expected_input_range_set = true;
  return *this;
}

SparseNetBuilder& SparseNetBuilder::weight_initializer(shared_ptr<Weight_initializer> initializer){
  if(nullptr != initializer){
    arg_weight_initer = initializer;
    is_weight_initializer_set = true;
  }else{
    is_weight_initializer_set = false;
  }

  return *this;
}

SparseNetBuilder& SparseNetBuilder::arena_ptr(google::protobuf::Arena* arena){
  arg_arena = arena;
  return *this;
}

SparseNetBuilder& SparseNetBuilder::allowed_transfer_functions_by_layer(vector<vector<transfer_functions> > filter){
  arg_allowed_transfer_functions_by_layer = filter;
  is_allowed_transfer_functions_by_layer_set = true;
  return *this;
}

SparseNetBuilder& SparseNetBuilder::weight_table(vector<sdouble32> table){
  if(0 < table.size()){
    arg_weight_table = table;
    is_weight_table_set = true;
  }else{
    is_weight_table_set = false;
  }
  return *this;

}

void SparseNetBuilder::set_weight_table(SparseNet* net){
  if(0 < arg_weight_table.size()){
    *net->mutable_weight_table() = {arg_weight_table.begin(), arg_weight_table.end()};
  }else throw "Unable to build net, weight table is of size 0!";
}

SparseNetBuilder& SparseNetBuilder::neuron_array(vector<Neuron> arr){
  /*! #2 */
  if((0 < arr.size())&&(Neuron_info::is_neuron_valid(arr.back()))){
    arg_neuron_array = arr;
    is_neuron_array_set = true;
  }else{
    is_neuron_array_set = false;
  }
  return *this;
}

void SparseNetBuilder::set_neuron_array(SparseNet* net){
  /*! #2 */
  if(Neuron_info::is_neuron_valid(arg_neuron_array.back())){ /* If the last element is valid */
    *net->mutable_neuron_array() = {arg_neuron_array.begin(),arg_neuron_array.end()};
  } else throw "Unable to set Neuron Array into Sparse net as the last Neuron seems invalid!";
}

SparseNet* SparseNetBuilder::denseLayers(vector<uint32> layer_sizes, vector<vector<transfer_functions> > filter){
  (void)allowed_transfer_functions_by_layer(filter);
  return denseLayers(layer_sizes);
}

SparseNet* SparseNetBuilder::denseLayers(vector<uint32> layer_sizes){
  
  using std::make_shared;

  uint32 previous_size = 0;
  uint32 numNeurons = 0;
  Synapse_interval temp_synapse_interval;
  /* Calculate number of weights needed overall
   * - Input Layer shall have a weight for every input for every neuron
   * - Input Layer shall have a weight for every bias and memory_filter for every neuron
   */
  uint64 numWeights = (layer_sizes[0] * arg_input_size) + (2 * layer_sizes[0]); /* The first layer only takes input from the @SparseNet input data */
  for(uint32 layerSize : layer_sizes){
    numNeurons += layerSize; /* Calculate the number of elements needed */
    numWeights += previous_size * layerSize; /* Calculate the number of weights needed */
    numWeights += layerSize * 2; /* Every neuron shall store its bias and memory_filter amongst the weights */
    previous_size = layerSize;
  }

  if(
    (is_input_size_set && is_expected_input_range_set ) /* needed arguments are set */
    &&(
      (!is_output_neuron_number_set) /* Output size is either not set */
      ||(arg_output_neuron_number == layer_sizes.back()) /* Or compliant to the Dense layer */
    )
  ){ /*! #3 */
    SparseNet* ret(google::protobuf::Arena::CreateMessage<SparseNet>(arg_arena));
    uint32 layerStart = 0;
    uint64 weightIt = 0;
    uint64 neurIt = 0;
    sdouble32 expPrevLayerOutput = Transfer_function::get_average_output_range(TRANSFER_FUNCTION_IDENTITY);

    ret->set_input_data_size(arg_input_size);
    ret->set_output_neuron_number(layer_sizes.back());

    previous_size = arg_input_size;

    if(!is_weight_initializer_set){
      weight_initializer(std::make_shared<Dense_net_weight_initializer>());
    }

    arg_weight_table = vector<sdouble32>(numWeights);
    arg_neuron_array = vector<Neuron>(numNeurons);

    previous_size = arg_input_size;
    for(uint32 layerIt = 0; layerIt < layer_sizes.size(); layerIt++)
    { /* Create the Dense Layers */

      /* Configuring the weight_initializerializer for this layer */
      arg_weight_initer->set(
        (0 == layerIt)?(arg_input_size):(layer_sizes[layerIt-1]),
        expPrevLayerOutput
      );

      /* Add the Neurons */
      expPrevLayerOutput = 0;
      for(uint32 layerNeurIt = 0; layerNeurIt < layer_sizes[layerIt]; layerNeurIt++){ 
        arg_weight_table[weightIt] = arg_weight_initer->next_bias();
        arg_weight_table[weightIt+1] = arg_weight_initer->next_memory_filter();
        arg_neuron_array[neurIt].set_bias_idx(weightIt);
        arg_neuron_array[neurIt].set_memory_filter_idx(weightIt+1);
        weightIt += 2;
        if(is_allowed_transfer_functions_by_layer_set){
          arg_neuron_array[neurIt].set_transfer_function_idx(
            Transfer_function::next(arg_allowed_transfer_functions_by_layer[layerIt])
          );
        }else{
          arg_neuron_array[neurIt].set_transfer_function_idx(Transfer_function::next());
        }

        /* Storing the expected output of this Net */
        if(0 < layerIt)expPrevLayerOutput += Transfer_function::get_average_output_range(
          arg_neuron_array[neurIt].transfer_function_idx()
        );

        /* Add the previous layer as an input synapse */
        temp_synapse_interval.set_starts(weightIt);
        temp_synapse_interval.set_interval_size(previous_size);
        *arg_neuron_array[neurIt].add_input_weights() = temp_synapse_interval;

        if(0 == layerIt){
          temp_synapse_interval.set_starts(Synapse_iterator::synapse_index_from_input_index(0));
        }else{
          temp_synapse_interval.set_starts(layerStart);
        }
        temp_synapse_interval.set_interval_size(previous_size);
        *arg_neuron_array[neurIt].add_input_indices() = temp_synapse_interval;

        for(uint32 neuron_weight_iterator = 0; neuron_weight_iterator < previous_size; neuron_weight_iterator++)
        { /* Fill in some initial neuron input values */
          arg_weight_table[weightIt] = arg_weight_initer->next_weight_for(
            arg_neuron_array[neurIt].transfer_function_idx()
          );
          weightIt++;
        }
        neurIt++; /* Step the neuron iterator forward */
      }

      if(0 == layerIt){
        expPrevLayerOutput = arg_expected_input_range;
        layerStart = 0;
      }else{
        expPrevLayerOutput /= static_cast<sdouble32>(layer_sizes[layerIt]);
        layerStart += previous_size;
      }
      previous_size = layer_sizes[layerIt];
    } /* Itearte through all of the layers */ 

    set_weight_table(ret);
    set_neuron_array(ret);
    return ret;
  }else throw "Input Output Pre-requisites failed;Unable to determine Net Structure!";
}

SparseNet* SparseNetBuilder::build(void){
  if(
    (is_input_size_set && is_output_neuron_number_set)
    &&(is_neuron_array_set && is_weight_table_set) /* needed arguments are set */
    &&(0 < arg_weight_table.size())&&(0 < arg_neuron_array.size()) /* There are at least some Neurons and Weights */
    &&(arg_output_neuron_number <= arg_neuron_array.size()) /* Output size isn't too big */
  ){
    SparseNet* ret = google::protobuf::Arena::CreateMessage<SparseNet>(arg_arena);
    ret->set_input_data_size(arg_input_size);
    ret->set_output_neuron_number(arg_output_neuron_number);
    set_weight_table(ret);
    set_neuron_array(ret);
    return ret;
  }else throw "Inconsistent parameters given to Sparse Net Builder!";
}

} /* namespace sparse_net_library */

#include <time.h>

#include <iostream>

#include "services/sparse_net_builder.h"

#include "models/dense_net_weight_initializer.h"
#include "services/synapse_iterator.h"

namespace sparse_net_library {

using std::shared_ptr;

SparseNetBuilder::SparseNetBuilder(){}

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
  if((0 < arr.size())&&(is_neuron_valid(&arr.back()))){
    arg_neuron_array = arr;
    is_neuron_array_set = true;
  }else{
    is_neuron_array_set = false;
  }
  return *this;
}

void SparseNetBuilder::set_neuron_array(SparseNet* net){
  /*! #2 */
  if(is_neuron_valid(&arg_neuron_array.back())){ /* If the last element is valid */
    *net->mutable_neuron_array() = {arg_neuron_array.begin(),arg_neuron_array.end()};
  } else throw "Unable to set Neuron Array into Sparse net as the last Neuron seems invalid!";
}

SparseNet* SparseNetBuilder::denseLayers(vector<uint32> layer_sizes, vector<vector<transfer_functions> > filter){
  (void)allowed_transfer_functions_by_layer(filter);
  return denseLayers(layer_sizes);
}

SparseNet* SparseNetBuilder::denseLayers(vector<uint32> layer_sizes){
  uint32 prevSize = 0;
  uint32 numNeurons = 0;
  /* Calculate number of weights needed overall
   * - Input Layer shall have a weight for every input for every neuron
   * - Input Layer shall have a weight for every bias and memory_ratio for every neuron
   */
  uint64 numWeights = (layer_sizes[0] * arg_input_size) + (2 * layer_sizes[0]); /* The first layer only takes input from the @SparseNet input data */
  for(uint32 layerSize : layer_sizes){
    numNeurons += layerSize; /* Calculate the number of elements needed */
    numWeights += prevSize * layerSize; /* Calculate the number of weights needed */
    numWeights += layerSize * 2; /* Every neuron shall store its bias and memory_ratio amongst the weights */
    prevSize = layerSize;
  }

  if(
    (io_pre_requisites_set() && is_expected_input_range_set ) /* needed arguments are set */
    &&(arg_output_neuron_number<=numNeurons) /* Output size isn't too big  */
  ){ /*! #3 */
    SparseNet* ret(google::protobuf::Arena::CreateMessage<SparseNet>(arg_arena));
    uint32 layerStart = 0;
    uint64 weightIt = 0;
    uint64 neurIt = 0;
    sdouble32 expPrevLayerOutput = Transfer_function_info::get_average_output_range(TRANSFER_FUNCTION_IDENTITY);

    ret->set_input_data_size(arg_input_size);
    ret->set_output_neuron_number(arg_output_neuron_number);

    prevSize = arg_input_size;

    if(!is_weight_initializer_set){
      weight_initializer(shared_ptr<Dense_net_weight_initializer>(new Dense_net_weight_initializer()));
    }

    arg_weight_table = vector<sdouble32>(numWeights);
    arg_neuron_array = vector<Neuron>(numNeurons);

    prevSize = arg_input_size;
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
        arg_weight_table[weightIt+1] = arg_weight_initer->next_memory_ratio();
        arg_neuron_array[neurIt].set_bias_idx(weightIt);
        arg_neuron_array[neurIt].set_memory_ratio_idx(weightIt+1);
        weightIt += 2;
        if(is_allowed_transfer_functions_by_layer_set){
          arg_neuron_array[neurIt].set_transfer_function_idx(
            Transfer_function_info::next(arg_allowed_transfer_functions_by_layer[layerIt])
          );
        }else{
          arg_neuron_array[neurIt].set_transfer_function_idx(Transfer_function_info::next());
        }

        /* Storing the expected output of this Net */
        if(0 < layerIt)expPrevLayerOutput += Transfer_function_info::get_average_output_range(
          arg_neuron_array[neurIt].transfer_function_idx()
        );

        /* Add the previous layer as an input synapse */
        arg_neuron_array[neurIt].add_input_index_sizes(prevSize);
        if(0 == layerIt)arg_neuron_array[neurIt].add_input_index_starts(Synapse_iterator::synapse_index_from_input_index(0));
          else arg_neuron_array[neurIt].add_input_index_starts(layerStart);
        for(uint32 neuron_weight_iterator = 0; neuron_weight_iterator < prevSize; neuron_weight_iterator++)
        { /* Fill in some initial neuron input values */
          arg_weight_table[weightIt] = arg_weight_initer->next_weight_for(
            arg_neuron_array[neurIt].transfer_function_idx()
          );
        
          arg_neuron_array[neurIt].add_weight_index_sizes(1);
          arg_neuron_array[neurIt].add_weight_index_starts(weightIt);
          weightIt++;
        }
        neurIt++; /* Step the neuron iterator forward */
      }

      if(0 == layerIt){
        expPrevLayerOutput = arg_expected_input_range;
        layerStart = 0;
      }else{
        expPrevLayerOutput /= static_cast<sdouble32>(layer_sizes[layerIt]);
        layerStart += prevSize;
      }
      prevSize = layer_sizes[layerIt];
    } /* Itearte through all of the layers */ 

    set_weight_table(ret);
    set_neuron_array(ret);
    return ret;
  }else throw "Input Output Pre-requisites failed;Unable to determine Net Structure!";
}

SparseNet* SparseNetBuilder::build(void){
  if(
    (io_pre_requisites_set())
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

bool SparseNetBuilder::is_neuron_valid(Neuron const * neuron)
{
  if(nullptr != neuron){
    if(
      (transfer_functions_IsValid(neuron->transfer_function_idx())) /* Transfer Function ID is valid */
      &&(TRANSFER_FUNCTION_UNKNOWN < neuron->transfer_function_idx()) /* Transfer Function ID is known */
      &&(( /* Either the input is consistent */
        (0 < neuron->input_index_sizes_size()) /* There are input index synapses */
        &&(neuron->input_index_sizes_size() == neuron->input_index_starts_size()) /* Index synapse is consistent */
        &&(0 < neuron->input_index_sizes(0)) /* Of non-null size */
        &&(0 < neuron->weight_index_sizes_size()) /* There are some weight synapses */
        &&(neuron->weight_index_sizes_size() == neuron->weight_index_starts_size()) /* Weight synapse is consitent */
        &&(0 < neuron->weight_index_sizes(0)) /* Of non-null size */
      )||( /* Or there is no input. we won't judge. */
        (0 == neuron->input_index_sizes_size()) && (0 == neuron->weight_index_sizes_size()) 
      ))
    ){ /*!Note: Only the first synapse sizes are checked for non-zero size for perfomance purposes. 
        * It is enough to determine if there is any input to the Neuron, because
        * if the first is non-zero then essentially there are more, than 0 inputs.
        */
      
      uint32 number_of_input_indexes = 0;
      for(int i = 0; i<neuron->input_index_sizes_size(); ++i){
        number_of_input_indexes += neuron->input_index_sizes(i);
      }

      uint32 number_of_input_weights = 0;
      for(int i = 0; i<neuron->weight_index_sizes_size(); ++i){
        number_of_input_weights += neuron->weight_index_sizes(i);
      }

      /* Check if inputs from synapses match */
      return (number_of_input_indexes == number_of_input_weights);
    } else return false;

  }else return false;
}

bool SparseNetBuilder::io_pre_requisites_set(void) const{
   return  (is_input_size_set && is_output_neuron_number_set);
}

} /* namespace sparse_net_library */

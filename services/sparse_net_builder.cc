#include <time.h>

#include "sparse_net_builder.h"
#include "models/dense_net_weight_initializer.h"

namespace sparse_net_library {

SparseNetBuilder::SparseNetBuilder(){}

SparseNetBuilder& SparseNetBuilder::input_size(uint32 size){
  arg_input_size = size;
  is_input_size_set = true;
  return *this;
}

SparseNetBuilder& SparseNetBuilder::input_neuron_size(uint32 num){
  arg_input_neuron_number = num;
  is_input_neuron_size_set = true;
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

SparseNetBuilder& SparseNetBuilder::weight_initializer(std::shared_ptr<Weight_initializer> initializer){
  if(nullptr != initializer){
    arg_weight_initer = initializer;
    is_weight_initializer_set = true;
  }else{
    is_weight_initializer_set = false;
  }

  return *this;
}

SparseNetBuilder& SparseNetBuilder::arena_ptr(google::protobuf::Arena* arena){
  if(nullptr != arena){
    arg_arena = arena;
    is_arena_ptr_set = true;
  }else{
    is_arena_ptr_set = false;
  }

  return *this;
}

SparseNetBuilder& SparseNetBuilder::weight_table(std::vector<sdouble32> table){
  /*! #5 */
  if(0 < table.size()){
    arg_weight_table = table;
    is_weight_table_set = true;
  }else{
    is_weight_table_set = false;
  }
  return *this;

}

void SparseNetBuilder::set_weight_table(SparseNet* net){
  /*! #5 */
  if(0 < arg_weight_table.size()){
    net->clear_weight_table();
    for (sdouble32& weight_table_element : arg_weight_table) {
      net->add_weight_table(weight_table_element);
    }
  }
}

SparseNetBuilder& SparseNetBuilder::neuron_array(std::vector<Neuron> arr){
  /*! #2 *//*! #5 */
  if((0 < arr.size())&&(neuronValid(&arr.back()))){
    arg_neuron_array = arr;
    is_neuron_array_set = true;
  }else{
    is_neuron_array_set = false;
  }
  return *this;
}

void SparseNetBuilder::set_neuron_array(SparseNet* net){
  /*! #2 *//*! #5 *//*! #6 */
  if(neuronValid(&arg_neuron_array.back())){ /* If the last element is valid */
    net->clear_neuron_array();
    for (Neuron& neuron_element : arg_neuron_array) {
      *net->add_neuron_array() = neuron_element;
    }
  }
}

SparseNet* SparseNetBuilder::denseLayers(std::vector<uint32> layerSizes, std::vector<std::vector<transfer_functions> > allowedTrFunctionsByLayer){
  uint32 prevSize = 0;
  uint32 numNeurons = 0;
  /* Calculate number of weights needed overall
   * - Input Layer shall have a weight for every input for every neuron
   * - Input Layer shall have a weight for every bias and memory_ratio for every neuron
   */
  uint64 numWeights = (arg_input_neuron_number * arg_input_size) + (2 * arg_input_neuron_number);
  for(uint32 layerSize : layerSizes){
    numNeurons += layerSize; /* Calculate the number of elements needed */
    numWeights += prevSize * layerSize; /* Calculate the number of weights needed */
    numWeights += layerSize * 2; /* Every neuron shall store its bias and memory ration amongst the weights */
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
    sdouble32 expPrevLayerOutput = Transfer_function_info::getAvgOutRange(TRANSFER_FUNC_IDENTITY);

    ret->set_input_data_size(arg_input_size);
    ret->set_input_neuron_number(arg_input_neuron_number);
    ret->set_output_neuron_number(arg_output_neuron_number);

    prevSize = arg_input_size;

    if(!is_weight_initializer_set){
      weight_initializer(std::shared_ptr<Dense_net_weight_initializer>(
         new Dense_net_weight_initializer(static_cast<uint32>(time(nullptr)))
       ));
    }

    arg_weight_table = std::vector<sdouble32>(numWeights);
    arg_neuron_array = std::vector<Neuron>(numNeurons);

    prevSize = arg_input_size;
    for(uint32 layerIt = 0; layerIt < layerSizes.size(); layerIt++)
    { /* Create the Dense Layers */

      /* Configuring the weight_initializerializer for this layer */
      arg_weight_initer->set(layerSizes[layerIt],expPrevLayerOutput);

      expPrevLayerOutput = 0;
      for(uint32 layerNeurIt = 0; layerNeurIt < layerSizes[layerIt]; layerNeurIt++)
      { /* Add the Neurons */
        arg_weight_table[weightIt] = arg_weight_initer->nextBias();
        arg_weight_table[weightIt+1] = arg_weight_initer->nextMemRatio();
        arg_neuron_array[neurIt].set_bias_idx(weightIt);
        arg_neuron_array[neurIt].set_memory_ratio_idx(weightIt+1);
        weightIt += 2; /*! #4 */
        arg_neuron_array[neurIt].set_transfer_function_idx(
          Transfer_function_info::next(allowedTrFunctionsByLayer[layerIt])
        );

        /* Storing the expected output of this Net */
        if(0 < layerIt)expPrevLayerOutput += Transfer_function_info::getAvgOutRange(
          arg_neuron_array[neurIt].transfer_function_idx()
        );

        /* Add the previous layer as an input partition */
        arg_neuron_array[neurIt].add_input_idx(layerStart);
        for(uint32 neuron_weight_iterator = 0; neuron_weight_iterator < prevSize; neuron_weight_iterator++)
        { /* Fill in some initial neuron input values */
          arg_weight_table[weightIt] = arg_weight_initer->nextWeightFor(
            arg_neuron_array[neurIt].transfer_function_idx()
          );
          arg_neuron_array[neurIt].add_input_weight_idx(weightIt);
          weightIt++; /*! #4 */
        }
        neurIt++; /* Step the neuron iterator forward */
      }

      if(0 == layerIt){
        expPrevLayerOutput = arg_expected_input_range;
        layerStart = 0;
      }else{
        expPrevLayerOutput /= static_cast<sdouble32>(layerSizes[layerIt]);
        layerStart += prevSize;
      }

      prevSize = layerSizes[layerIt];
    }

    set_weight_table(ret);
    set_neuron_array(ret);
    return ret;
  }else{
    throw INVALID_BUILDER_USAGE_EXCEPTION;
  }
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
    ret->set_input_neuron_number(arg_input_neuron_number);
    ret->set_output_neuron_number(arg_output_neuron_number);
    set_weight_table(ret);
    set_neuron_array(ret);
    return ret;
  }else throw INVALID_BUILDER_USAGE_EXCEPTION;
}

bool SparseNetBuilder::neuronValid(Neuron const * neuron) const
{
  if(nullptr != neuron){
    return (
      (transfer_functions_IsValid(neuron->transfer_function_idx())) /* Transfer Function ID is valid */
      &&(0 < neuron->input_idx_size()) /* There are input idexes */
      &&(0 < neuron->input_weight_idx_size()) /* There are some connection weights */
    );
  }else return false;
}

bool SparseNetBuilder::io_pre_requisites_set(void) const{
   return  (is_input_size_set && is_input_neuron_size_set && is_output_neuron_number_set);
}

} /* namespace sparse_net_library */

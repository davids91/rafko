#ifndef sparse_net_global_H
#define sparse_net_global_H

namespace sparse_net_library {

/***
 * TODO:
 *  - Add Output Normalizer to Neuron (?)
 *  - Store vanishing gradient / exploding gradient statistics based on indexes
 *  - Look out for https://stackoverflow.com/questions/26960906/protobuf-repeated-fields-deserialization
 *  - #1 Include License information for every File
 *  - #2 in SparseNetBuilder::neuron_array validate the whole of the array if possible
 *  - #3 Set Arena options for memory usage
 *  - #4 Finish Generating the solution chain from the SparseNet
 *  - #5 use `make_unique` everywhere instead of `new`
 *  - #6 use references instead of pointers where applicable
 *  - #7 use Exception Class with String formatter  
 *  - #8 more complex Partial Solution test
 *  - #9 Rule of Three: Destructor, Copy Constructor, Copy assignment operator should bedefined for every relevant class
 *  - #10 use uint64 where needed! ( especially because of the iteration logic )
 *  - #12 test transfer function info
 *  - #13 use the protobuffer version verifier
 *  - #14 use [[deprecated("msg")]] - when applicable
 */
/**
 * GLOBAL TYPES
 */
typedef unsigned long uint64;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;
typedef signed long sint64;
typedef signed int sint32;
typedef signed short sint16;
typedef signed char sint8;
typedef double sdouble32;
typedef uint16* p_uint16;
typedef sdouble32* p_sdouble32;

} /* namespace sparse_net_library */
#endif /* defined sparse_net_global_H */

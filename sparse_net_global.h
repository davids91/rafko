#ifndef sparse_net_global_H
#define sparse_net_global_H

namespace sparse_net_library {

/***
 * TODO:
 *  - Add Output Normalizer to Neuron (?)
 *  - SparseNet Transfer Function initialization
 *  - Include License information for every File
 *  - Store vanishing gradient / exploding gradient statistics based on indexes
 *  - Look out for https://stackoverflow.com/questions/26960906/protobuf-repeated-fields-deserialization
 *  - #1 srand is in effect for the whole application, this might not be safe
 *  - #2 in SparseNetBuilder::neuron_array validate the whole of the array if possible
 *  - #3 Set Arena options for memory usage
 *  - #4 eliminate local variable
 *  - #5 test vigorously, especially for memory leaks
 *  - #6 Use mutable, to add the whole array, don't use a cycle
 *  - #8 Extend testcases with biases
 *  - #9 Test neuron valid interface in builder
 *  - #12 test transfer function info
 *  - #13 use the protobuffer version verifier
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

/**
 * Exceptions
 */
typedef enum {
  NOT_IMPLEMENTED_EXCEPTION,
  INVALID_USAGE_EXCEPTION,
  NULL_DETAIL_EXCEPTION,
  UNIDENTIFIED_OPERATION_EXCEPTION
}sparse_net_library_exception;

} /* namespace sparse_net_library */
#endif /* defined sparse_net_global_H */

#ifndef sparse_net_global_H
#define sparse_net_global_H

namespace sparse_net_library {

/***
 * Future Features:
 *  - Add Output Normalizer to Neuron (?)
 *  - Store vanishing gradient / exploding gradient statistics based on indexes
 *  - Look out for https://stackoverflow.com/questions/26960906/protobuf-repeated-fields-deserialization
 *  - Implement uint64 usage to support giant nets
 * TODOs:
 *  - #1 Include License information for every File
 *  - #2 in SparseNetBuilder::neuron_array validate the whole of the array if possible
 *  - #3 Set Arena options for memory usage
 *  - #4 Remove actual_index from @Partial solution, substitue with @output_data synapse
 *  - #5 Implement weight sharing in @Partial_solution_builder (?)
 *  - #6 Implement Softmax activation function
 *  - #7 use Exception Class with String formatter
 *  - #8 Add Partial Solution tests to more complex structures as well
 *  - #9 Implement and test Gradient calculation
 *  - #10 Implement the protocol buffer server
 *  - #11 Implement Regularization ( L1, L2 )
 *  - #12 test transfer function info
 *  - #13 use the protobuffer version verifier
 *  - #14 make make paralell by -j
 *  - #15 Implement Disentanglement
 *  - #16  Implement Calcium excitation: A value is added for each Neuron input: the difference from the other inputs ( ? simulate XOR with 1 Neuron somehow? )
 *  - #17 - neuron memory Spike function: f(prev, curr, neuron_param) = prev + (prev - curr)* neuron_param
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
 * GLOBAL PARAMETERS
 *//* Later to be migrated to the server implementation.. */
extern sdouble32 epsilon; /* very small positive value almost greater, than 0.0 */
extern sdouble32 lambda;
extern sdouble32 alpha;

} /* namespace sparse_net_library */
#endif /* defined sparse_net_global_H */

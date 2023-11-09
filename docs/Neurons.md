# Neuron Structure


Each Neuron has a standard structure.

While the weights might be taken from all over the networks weight table in an unforseeable way, the index values of them are always defined by the same structure in the neuron definition. A @Neuron with `n` input and (`1` + `n` + `m`) weights looks like the following:

| Neuron structure: | ws | w1 | .. | wn | bias1 | .. | biasm |
|------------------|----|----|----|----|-------|----|-------|

Inside the &Neuron structure only weight index values (`std::uint32_t`) are stored, which points to elements inside the networks weight table. The first weight is the weight of the @Neurons spike function, the following `n` weights represent one weight for every input. The remaining weights are all used as bias values.

An example of what that might look like:

| Network Weight table | w|  ..weights_for Neuron1.. |w| .. weights for Neuron2 .. | w |
|----------------------|--|--------------------------|-|---------------------------|---|

This way weights are re-usable by structure and there is a possibility for every Neuron to have multiple and/or shared bias values.

A @Solution and multiple @PartialSolution objects are Based on this structure. Each @PartialSolution however has a different weight table exclusive to the neurons inside.

The Structure of a @Neuron inside a @PartialSolution is the same, despite multiple neuron input and weight information being stored in a single array.

## Forward propagation Neuron-by-Neuron

Below is a summary of the propagation steps, each column contains either required data for propagation or placeholders to help in data serialization.

| Step Desicription       | Weight data         | Dependency data     | Dependency data     | Dependency data     | Backprop op index   | Behavior data                     |
|-------------------------|---------------------|---------------------|---------------------|---------------------|---------------------|-----------------------------------|
| Neuron Bias             | weight_index        | is_there_dependency | dependency_index    |                     | operation_index     | input_function_index(optional)    |
| Neuron input            | weight_index        | dependency_index    | dependency_index    | input_count         | operation_index     | input_function_index + past_index |
| Objective operation     |                     |                     |                     |                     |                     |                                   |
| Solution operation      |                     | neuron_index_start  |                     | neuron_count        |                     |                                   |
| Spike Function          | weight_index        |                     | dependency_index    |                     | operation_index     | spike_function_index              |
| Transfer function       |                     |                     | dependency_index    |                     | operation_index     | transfer_function_index           |
| Weight regularization   |                     |                     |                     |                     |                     |                                   |

Each step collects some information from its input buffers and stores the result in a target buffer; The operations are based on the Structure of the Neuron (e.g.: what kind of transfer function or spike function to use); and the result is stored depending on the action.

In case input function is taking its input from the past, the information is arriving through a combined bitstring which contains both the input function index ( lower half ) and the past index ( upper half ); so both the number of input functions and the number of memory slots are maximized to 15.

Note: Since for One Neurons might have multiple Bias values, the information must be stored within the table should there be additional bias dependency for the exact operator. Should this be the case, dependency_index and input function index is relevant.


## Back-propagation Operation-by-Operation

| Step Desicription             | Weight data         | Dependency data           | Dependency data     | Dependency data     |  Backprop op index  | Behavior data                     |
|-------------------------------|---------------------|---------------------------|---------------------|---------------------|---------------------|-----------------------------------|
| d/dw of Bias(es)              | weight_index        | is_there_dependency       | dependency_index    |                     | operation_index     | input_function_index(optional)    |
| d/dw of Neuron input          | weight_index        | dependency_index          | dependency_index    |                     | operation_index     | input_function_index + past_index |
| d/dw of Objective operation   | label_index         | sample_number             | dependency_index    |                     | operation_index     | cost_function_index               |
| d/dw of Solution operation    |                     |                           |                     |                     |                     |                                   |
| d/dw of Spike Function        | weight_index        |                           | dependency_index    |                     | operation_index     | spike_function_index              |
| d/dw of Transfer function     |                     |                           | dependency_index    |                     | operation_index     | transfer_function_index           |
| d/dw of Weight regularization | weight_index_start  |                           |                     | count_weights       | operation_index     | feature_index                     |

## Consolidated Instruction table

Each instruction for Network inference and error back-propagation with thier respective requirements are summarized in the above 2 tables. Because of the similarities, they can be merged into one final table; Whether or not it's derivative shall be decided by the context and the execution state of the GPU Kernel.

| Step Desicription       | Input Global Buffer | Input Global Buffer | Input Global Buffer     | Input Global Buffer | Behavior Index                    |
|-------------------------|---------------------|---------------------|-------------------------|---------------------|-----------------------------------|
| Neuron Bias             | weight_index        |                     | dependency_descriptor*  | operation_index     | input_function_index(optional)    |
| Neuron input            | weight_descriptor*  | dependency_index    | dependency_index        | operation_index     | input_function_index + past_index |
| Objective operation     | label_index         | sample_number       | dependency_index        | operation_index     | cost_function_index               |
| Solution operation      | neuron_index_start  | count_neurons       |                         |                     |                                   |
| Spike Function          | weight_index        |                     | dependency_index        | operation_index     | spike_function_index              |
| Transfer function       |                     |                     | dependency_index        | operation_index     | transfer_function_index           |
| Weight regularization   | weight_index_start  | count_weights       |                         | operation_index     | feature_index                     |

\*weight_descriptor, dependency_descriptor either contains the used index, or a special value meaning: not used
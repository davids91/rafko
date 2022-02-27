# Neuron Structure


Each Neuron has a standard structure.

While the weights might be taken from all over the networks weight table in an unforseeable way, the index values of them are always defined by the same structure in the neuron definition. A @Neuron with `n` input and (`1` + `n` + `m`) weights looks like the following:

| Neuron structure: | ws | w1 | .. | wn | bias1 | .. | biasm |
|------------------|----|----|----|----|-------|----|-------|

Inside the &Neuron structure only weight index values (`std::uint32_t`) are stored, which points to elements inside the networks weight table. The first weight is the weight of the @Neurons spike function, the following `n` weights represent one weight for every input. The remaining weights are all used as bias values.

An example of what that might look like:

| Network Weight table |   |   | bias1 |   | ws |   | w1 | w2 | bias2 | .. | biasm | w3 | .. | wn |
|----------------------|---|---|-------|---|----|---|----|----|-------|----|-------|----|----|----|

This way weights are re-usable by structure and there is a possibility for every Neuron to have multiple and/or shared bias values.

A @Solution and multiple @PartialSolution objects are Based on this structure. Each @PartialSolution however has a different weight table exclusive to the neurons inside.

The Structure of a @Neuron inside a @PartialSolution is the same, despite multiple neuron input and weight information being stored in a single array.

# Neuron behavior

Neurons in a GPU Kernel behave based on the below script:

<table><thead><tr><th colspan="2">Step Description</th><th colspan="2">Input global buffer</th><th>Behavior index</th><th>Action</th></tr></thead><tbody><tr><td>1.</td><td>Collect first input from global data, multiply it by the assigned weight and store it inside the local buffer</td><td>Weight</td><td>Input/neuron_data</td><td></td><td>set_buffer</td></tr><tr><td>2. </td><td>Continue the input collection, update local buffer with the result of the update function; Repeat until the end of the inputs</td><td>Weight</td><td>Input/neuron_data</td><td>input_function_index</td><td>input_function</td></tr><tr><td>5.</td><td>Collect each remaining weight with the input function as bias, each time updating the local buffer</td><td>Weight</td><td></td><td>input_function_index</td><td>input_function</td></tr><tr><td>6. </td><td>Apply the given transfer function to the collected inputs, store the result in the local buffer</td><td></td><td>Input/neuron_data</td><td>transfer_function_index</td><td>transfer_function</td></tr><tr><td>7.</td><td>Apply Spike function to the result of the transfer function, store the result in the global pointer</td><td>Weight</td><td>neuron_data</td><td>spike_function_index</td><td>spike_function</td></tr></tbody></table>

Each step collects some information from its input buffers and stores the result in a target buffer; Where the operations are based on the Structure of the Neuron (e.g.: what kind of transfer function or spike function to use); and the result is stored depending on the action.

 Most operations are stored on a local buffer and only the end result is stored in a global buffer. This is partly because of locality and aims to help cache optimization.

 The local and global memory naming in this description reflects the (OpenCL memory model)[https://www.khronos.org/registry/OpenCL/specs/2.2/html/OpenCL_API.html#_memory_model].

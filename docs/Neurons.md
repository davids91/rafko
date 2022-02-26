Neuron Structure
===

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

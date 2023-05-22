# Operation matrix

Described in Neurons.md a set of operations required for network inference and back-propagation. However to fully make use of the massive coure count in most modern GPU-s, 
many of these operations should run in paralell. The GPU Phase which also receives the dataset inputs and labels receives the Neural network data also contains all the operational data required for inference and training.

## Operation Matrix structure in theory

As the name implies, the operation matrix is a 2 dimensional construct in whcih the rows represent one batch of oeprations that can be executed in paralell. Columns of the structure might vary. One element in such a matrix should provide data
corresponding to the tables in Neurons.md forward or back-propagation sections. 
As each row can be executed in parallel, each item in a row might depend on items in previous rows, 
a synchronisation point is required between rows.

Example: 

|-----------------|-----------------|-----------------|-----------------|-----------------|
|     Operation   |     Operation   |     Operation   |     Operation   |     Operation   |
|-----------------|-----------------|-----------------|-----------------|-----------------|
|     Operation   |     Operation   |                 |                 |                 |
|-----------------|-----------------|-----------------|-----------------|-----------------|
|     Operation   |     Operation   |     Operation   |                 |                 |
|-----------------|-----------------|-----------------|-----------------|-----------------|
|     Operation   |     Operation   |     Operation   |     Operation   |                 |
|-----------------|-----------------|-----------------|-----------------|-----------------|
|     Operation   |     Operation   |     Operation   |     Operation   |     Operation   |
|-----------------|-----------------|-----------------|-----------------|-----------------|

In the above table, the minimum number of operations whcih can be executed in parallel are 2. 

## Operation matrix structure in GPU-compatible format

Because the sizes of each row might vary, an additional number is required to be added in order to know where are the 
required synchronisation points (between rows).

|---|-----------------|-----------------|-----------------|-----------------|-----------------|
| 5 |     Operation   |     Operation   |     Operation   |     Operation   |     Operation   |
|---|-----------------|-----------------|-----------------|-----------------|-----------------|
| 2 |     Operation   |     Operation   |                 |                 |                 |
|---|-----------------|-----------------|-----------------|-----------------|-----------------|
| 3 |     Operation   |     Operation   |     Operation   |                 |                 |
|---|-----------------|-----------------|-----------------|-----------------|-----------------|
| 4 |     Operation   |     Operation   |     Operation   |     Operation   |                 |
|---|-----------------|-----------------|-----------------|-----------------|-----------------|
| 5 |     Operation   |     Operation   |     Operation   |     Operation   |     Operation   |
|---|-----------------|-----------------|-----------------|-----------------|-----------------|

Each operation contains a given number of bytes, but to simplify the handling of operations, despite potential different operation data(olike index values) sizes, all shall have uniform sizes constraining to the tables in Neurons.md forward or back-propagation sections. 
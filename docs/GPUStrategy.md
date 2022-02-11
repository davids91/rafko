# GPU Strategy


Different stages of Training Deep Networks require different, but structurally similar inputs.

## Deep Learning pipeline
### Forward propagation:
The network is reading in `n` number of inputs provided by the framework environment, and produces `m` number of features based on them. The inputs may represent  a sample, a sequence or any arbitrary form of data; The only requirement is that the inputs be divisible by the number of inputs the network is accepting. This makes sure the network can correctly cover every input in the vector. The output is proportional to the number of outputs the network is producing.

-  **Inputs:**
  - `n*in` number of floating point numbers mapped to a 1 dimensional array where `in` is the number of inputs the network accepts.
  - `v` number of floating point numbers corresponding to the networks weight_table

- **Outputs:**
  - `n*out` number of floating point numbers mapped to a 1 dimensional array where `out` is the feature size the network produces for one solve.

### Error Calculation step 1:
The framework is accepting `n*out` number of calculated features(network outputs) and compares them to the same number of labels, provided by the framework objective. The feature errors are stored in a 1 dimensional vector, and optionally a summation operation is done on them, resulting in a single compressed error value for the provided feature-label pair.

- **Inputs:**
  - `n*out` floating point numbers mapped to a 1 dimensional array where `out` is the number of inputs the network produces.
  - `n*out` floating point numbers, provided by the frameworks Environment, representing the expected outputs calculated from the corresponding input values

-  **Outputs:**
  - `n*out` number of floating point numbers mapped to a 1 dimensional array where `out` is the number of features the network produces; representing the per-feature error values.
  - A single number representing the overall error of the inputs, calculated from the per-feature error vector also being provided by the Error calculation step.

### Back-propagation:
Based on the calculated features, errors and the structure of the network, in the backwards pass of the network the gradients are calculated; Whom are used to update the network weights. In the current framework however, to let the networks have wild:tm: architectures( not always differentiable directly ), updating the weights is based on gradient approximation from errors instead of numerical gradients. Therefore the Back-propagation step is not currently supported in this framework; Maybe in the future, based on auto-diff.

-  **Inputs:**
  - `n*in` number of floating point numbers mapped to a 1 dimensional array where `in` is the number of **inputs the network accepts**.
  - `v` number of floating point numbers corresponding to **the networks weight_table** representing the weight values of the network.
  - `n*out` number of floating point numbers mapped to a 1 dimensional array, which is **the feature the network produces**.
  - `n*out` number of floating point numbers mapped to a 1 dimensional array where `out` is the number of features the network produces; representing **the per-feature error values**.
  - `n*out` floating point numbers, provided by the frameworks Environment, representing **the expected outputs** calculated from the corresponding input values


- **Outputs:**
  - `n*out` number of floating point numbers mapped to a 1 dimensional array where `out` is the feature size the network produces for one solve.
  - `v` number of floating point numbers corresponding to the networks weight_table representing the weight gradient values of the network.

## Generic GPU pipeline step
Based on the defined deep learning steps to attain error and gradient information a generalized computing structure can be defined as such:

<img align="left" src="../res/flow_arrow_down.png">
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow">input_1_0</th>
    <th class="tg-baqh">...</th>
    <th class="tg-baqh">input_1_*</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">input_n_0</td>
    <td class="tg-baqh">...</td>
    <td class="tg-baqh">input_n_*</td>
  </tr>
  <tr>
    <td class="tg-c3ow" colspan="3">Calculation Kernel</td>
  </tr>
  <tr>
    <td class="tg-baqh">output_0_0</td>
    <td class="tg-baqh">...</td>
    <td class="tg-baqh">output_0_*</td>
  </tr>
</tbody>
</table>
<img align="left" src="../res/flow_arrow_down.png">

## Generic GPU pipeline phase

A phase consists of multiple GPU pipeline steps with defined order. Each step has input is of the same dimensions as the output of the step preceding it, leaving only the first and last steps input to be freely defined. The following example implements a phase with 3 steps, each step takes the input of the preceding step.

<img align="left" src="../res/flow_arrow_down.png">
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow">input_1_0</th>
    <th class="tg-c3ow">...</th>
    <th class="tg-c3ow">input_1_x</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">input_n_0</td>
    <td class="tg-c3ow">...</td>
    <td class="tg-c3ow">input_n_y</td>
  </tr>
  <tr>
    <td class="tg-c3ow" colspan="3">Calculation Kernel 0</td>
  </tr>
  <tr>
    <td class="tg-c3ow">output_0_0</td>
    <td class="tg-c3ow">...</td>
    <td class="tg-c3ow">output_0_*</td>
  </tr>
  <tr>
    <td class="tg-baqh">output_1_0</td>
    <td class="tg-baqh">...</td>
    <td class="tg-baqh">output_1_*</td>
  </tr>
  <tr>
    <td class="tg-c3ow" colspan=3>Calculation Kernel 1</td>
  </tr>
  <tr>
    <td class="tg-c3ow">output_2_0</td>
    <td class="tg-c3ow">...</td>
    <td class="tg-c3ow">output_2_*</td>
  </tr>
  <tr>
    <td class="tg-c3ow" colspan="3">Calculation Kernel 2</td>
  </tr>
  <tr>
    <td class="tg-c3ow">output_3_0</td>
    <td class="tg-c3ow">...</td>
    <td class="tg-c3ow">output_3_*</td>
  </tr>
  <tr>
    <td class="tg-c3ow">output_4_0</td>
    <td class="tg-c3ow">...</td>
    <td class="tg-c3ow">output_4_*</td>
  </tr>
  <tr>
    <td class="tg-c3ow">output_5_0</td>
    <td class="tg-c3ow">...</td>
    <td class="tg-c3ow">output_5_*</td>
  </tr>
</tbody>
</table>
<img align="left" src="../res/flow_arrow_down.png">

Each GPU relevant Object inside the framework(Environment, Objective, Agent) shall implement this general structure to signal compatibility with GPU relevant calculations.

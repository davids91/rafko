# C++ - Sparse Net Library
This folder contains the backbone of the Neural network calculation library. 
The C++ parts are made to implement a server for Neural Network operations.
It shall ommunicate mainly with the Client implementations in other languages, 
providing services for: 

 - Neural network creation:
  - For testing the library throughly a simple Neural Network builder is provided, 
    which provides interfaces to build Neural networks from raw Neuron arrays, and a
    builder to produce Fully connected feedforward layers, with the possibility of a recurrent netwrok. Unfortunately LSTM and GRU and such is not supported due to the structure of the Network.
 - Neural network solution
  - From a Neural network definition (@Sparse_net) a matrix of partial solutions is generated, which is composed into a Solution object, which is solvable effectively for CPUs and GPUS as well. GPU support is planned through OpenCL, but currently it's not in focus.
 - Neural network optimization
  - A basic optimizer is provided [in the services folder](https://github.com/davids91/rafko/blob/master/cxx/services/src/sparse_net_optimizer.cc) which supports training for [Neural networks of non-circular nature](https://en.wikipedia.org/wiki/Tree_(graph_theory)). Training Recurrent networks is available through Truncated BPTT, but use with caution, as supporting traditional derivative based gradient calculations is not the focus of this framework. 
  - The main focus of the framework is to train networks of previously unseen architectures, so it oftentimes encounters network architectures in which the optimization is sub-optimal or incorrect due to the complicated formulae and mathematical structure of it. Because of this a gradient approximation method is to be used, in which the gradients are approximated, instead of calculated.

## Dependencies: 

 - pkg-config
 - protocol buffers (proto3 compatible)
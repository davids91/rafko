# Rafko [![Join the chat at https://gitter.im/rafko_deep_learning/community](https://badges.gitter.im/rafko_deep_learning/community.svg)](https://gitter.im/rafko_deep_learning/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

A deep learning Framework extended with per Neuron memory capabilities with focus on 
architecture search through training.
The Framework is of a server-client architecture, in which  a deep learning server provides 
calculation slots for different clients. The calculations are neural network related, 
such as solving a network, calculating gradients and updating a network, etc.. 

## Supported languages: <img align="right" src="res/logo_framed.png">
### Deep learning server
- cxx14

### Deep learning Client
- cxx ( planned )
- Java ( planned )
- Python ( planned )

## Folders:
 - **/build:** Contains the build script and also the generated objects after build 
 - **/cxx:** Contains the Source code for the Sparse Network Library
   - **gen:** protocol buffer generated files
   - **models:** the building blocks of the cxx Sparse net library server
   - **services:** objects providing some kind of service each
   - **test:** Unit tests based on the [Catch Framework](https://github.com/catchorg/Catch2) 
 - **/java:** Contains prototypes and the Client implementation
   - **sandboxes:** Prototypes for experiments
   - **client:** the Java Client library ( missing )
 - **/proto:**
   - **models:** The building blocks of the framework 
   - **services:** Defines the commands available for the client libraries ( missing )
 - **/res:** miscellaneous resources

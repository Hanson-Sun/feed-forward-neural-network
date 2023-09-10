# feed-forward-neural-network
 a little fun with FFNNs in C++. A basic implementation of a FFNN, as an exercise to learn ML along with c++. 
 This project also utilizes a multi-threaded matrix math library that I created.


 ## Matrix library

A lightweight cpp matrix library that is wrapped in the `Math` namespace (see `math.h` in `headers`). The code should be documented. Note, that math vectors are considered to be column matrices for cleaner code interoperability. It uses double precision (kinda excessive) and it supports several common operations used in linear algebra:
- Initialization with a single double `vector` will be considered a mathematical vector by default
- `*, -, +, /` and `*=, -=, +=, /=` operators between matrices and doubles
- `*, -, +` and `*=, -=, +=` between matrices
- matrix transpose, inner, outer, and Hadamard products
- a special dot product between two "vectors" (row dot row matrix or column dot column matrix)
- random matrix generation (default random engine, not optimized for better numerical distribution)
- to string serialization and direct `<<` overloading
- direct type casting into a double `vector`

A `ThreadPool` class is also baked in for any uses outside of the matrix library. Many matrix operations are multi-threaded for optimal performance.


## Feed Forward Neural Network
The FFNN consists of several components
- `FFNN` class
- `Activation::ActivationFn` different activation functions (internal methods need to be implemented)
- `Cost::CostFn` cost functions (internal methods also need to be implemented)
- A data reading class that is specifically built for testing

## Performance and Accuracy
to be updated.... once i have time to actually run tests


## Current issues
- (BUG) Terminate is called recursively once a certain amount of memory is reached
- can have further code optimizations

## Todo
- test FFNN with MNIST dataset
- figure out a direct matrix approach 
- automatic differentiation (might be really hard to implement...)
- optimize the way data is loaded
- implement cross-validation (done) and other proper analysis methods
- test out some variable learning rate algs (momentum, ADAM, ...)

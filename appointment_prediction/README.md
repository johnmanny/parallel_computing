# Appointment Prediction
This project aims to compare performance between parallel and serial implementations of a
neural network classifier used for prediction.

## Details
The network takes in 9 input variables from 'data/trimmedApptData.csv' and creates
a number of hidden layers, as well as neurons per layer, passed in from the execution.
It uses backpropogation to calculate the error and propogate required changes to the 
weights in previous network layers in order to account for the difference in example and
network output. 

## Use
The number of hidden layers and neurons in each respective layer are passed as arguments:
There are 2 layers which aren't modified. 
- The first layer will always have 9 input neurons
- The output (last) layer will always only have 1 neuron

```
make		   // compiles
make run	   // runs network automatically with 2 hidden layers - first with 5 neurons second with 6

./ser_main 5 6	   // run with 5 neurons in 1st hidden layer and 6 neurons in 2nd hidden layer
./ser_main 4	   // run with 4 neurons in 1 hidden layer
./ser_main 8 6 4 3 // run with 8 neurons in first hidden layer, 6 in second, 4 in 3rd, 3 in 4th

make clean	   // cleans compiled files
```
## Sources
TBD
More specific sources are cited in comments.

# Batch Neural Network Training in Parallel
This directory holds the batch training version of the custom network.
Runs in parallel with default 6 cores using OpenMP.

## Use
```
make		// compiles work-in-progress batch version 'batch_main'
make run	// runs work in progress batch_main
make clean	// cleans directory 
```

## Notes
- These files somewhat dirty but should be considered the final result of the 
prediction network. With default settings, the network predicts with around
65% accuracy with the data broken up in the defined way in the files (shuffled
after each training set runthrough, equal number of no/yes cases)

- The main network actually learns in a method that is more accurately described
as a mini-batch gradient descent method of supervised neural net learning.

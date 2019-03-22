# Batch Neural Network Training in Parallel
This directory holds the batch training version of the custom network.
Runs in parallel with default 6 cores using OpenMP.

## Use
```
make		// compiles work-in-progress batch version 'batch_main'
make runB	// runs work in progress batch_main
make batch_o	// runs old batch which guesses prediction from uneven examples
make runO	// runs old one
make clean	// cleans directory 
```

## Notes
These files are very, very dirty because of a time crunch. I will look to clean
it up soon. Also, currently does not have error checking and the work-in-progress
version still has bugs that are being worked out.

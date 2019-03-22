#Supplementary Project README
The original Yannl readme has been left untouched to properly credit the author.

##Testing
Note: ser_main.cpp was a dead-end work in progress. It has been lef to show the work that was done on it. We did not report on it.

There are two makefiles. The capital "M" makefile is for the Yannl Library that we have altered.
You can test the SIMD/Matrix paralellization by uncommenting the #pragma lines in matrix.cpp or network.cpp and running "sudo make" in the main directory.
To test the parallel code you have compiled, run "make -B" and then "make realistic_classification" in the examples folder. The number of features can be easily manipulated at the top of the file.
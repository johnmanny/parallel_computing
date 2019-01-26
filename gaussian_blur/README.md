# Gaussian blur
This repo implements gaussian blur using serial and parallel code utilizing 
intel's ISPC compiler for SIMD (single instruction multiple data) programming.

# Description
For both implementations, a fake 'border' around the image is used to account
for edge cases to avoid memory errors. Each pixel on the extra border
just has zeros for each RGB value.

During execution, a timer is used to display the approximate runtime in seconds
for each implementation.

# Use
Since this project was implemented for platform-specific specifications, the makefile
parallel targets likely won't run on every machine. 

According to project specifications, here are the makefile commands:
```
make 		// compiles serial implementation
make par	// compiles parallel implementation, loads ispc module
make runs	// runs serial implementation
make runp	// runs parallel implementation
make clean	// cleans everything - oh wow
```
# Acknowledgements
JPEG input, output, and other tasks accomplished with use of CImg library (CImg.h). All
other non-original work is cited.

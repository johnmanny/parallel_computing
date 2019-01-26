# gaussian blur
This repo implements gaussian blur using serial and parallel code utilizing 
intel's ISPC compiler for SIMD (single instruction multiple data) programming.

# description
For both implementations, a fake 'border' around the image is used to account
for special cases and out-of-bounds calculations. Each pixel on the extra border
just has zeros for each RGB value.

During execution, a timer is used to display the approximate runtime in seconds.

# use
While the makefile has some platform specific flags, use through the makefile 
is:
```
make 		// compiles serial implementation
make par	// compiles parallel implementation
make runs	// runs serial implementation
make runp	// runs parallel implementation
make clean	// cleans everything - oh wow
```

# Authors: John Nemeth
COMPILER= g++
CFLAGS= -g -Wall -std=c++11 -m64 -O3

# batch implementation attempt - john
batch_main: batch_main.cpp
	$(COMPILER) $(CFLAGS) -fopenmp batch_main.cpp -o batch_main

# Arguments are number of hidden neurons on each layer
#   I.E.: 5 neurons at first hidden layer, 6 neurons at second hidden layer,etc

run:
	./batch_main 5 4

clean:
	rm ser_main.o ser_main par_main par_main.o log batch_main batch_o

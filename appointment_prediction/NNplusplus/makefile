# Authors: John Nemeth
COMPILER= g++
CFLAGS= -g -Wall -std=c++11 -m64 -O2


ser_main: ser_main.o Matrix.o NeuralNet.o
	$(COMPILER) $(CFLAGS) ser_main.o Matrix.o NeuralNet.o -o ser_main

ser_main.o: ser_main.cpp
	$(COMPILER) $(CFLAGS) -c ser_main.cpp -o ser_main.o

Matrix.o: Matrix.cpp
	$(COMPILER) $(CFLAGS) -c Matrix.cpp -o Matrix.o

NeuralNet.o: NeuralNet.cpp
	$(COMPILER) $(CFLAGS) -c NeuralNet.cpp -o NeuralNet.o

par_main: par_main.o
	$(COMPILER) $(CFLAGS) -fopenmp par_main.o -o par_main

par_main.o: par_main.cpp
	$(COMPILER) $(CFLAGS) -fopenmp -c par_main.cpp -o par_main.o
#Nodes layers
# Arguments are number of nodes per layer, and they number of layers
# so 5 1 is one layer of 5 nodes
run:
	./ser_main 5 1

runP:
	./par_main 5 1

log:
	./ser_main > log

clean:
	rm ser_main.o ser_main par_main par_main.o log

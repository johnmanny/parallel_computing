# Authors: John Nemeth
COMPILER= g++
CFLAGS= -g -Wall -std=c++11 -m64 -O2


array_ser_main: array_ser_main.o
	$(COMPILER) $(CFLAGS) array_ser_main.o -o array_ser_main

array_ser_main.o: array_ser_main.cpp
	$(COMPILER) $(CFLAGS) -c array_ser_main.cpp -o array_ser_main.o

array_par_main: array_par_main.o
	$(COMPILER) $(CFLAGS) -fopenmp array_par_main.o -o array_par_main

array_par_main.o: array_par_main.cpp
	$(COMPILER) $(CFLAGS) -fopenmp -c array_par_main.cpp -o array_par_main.o

# Arguments are number of hidden neurons on each layer
#   I.E.: 5 neurons at first hidden layer, 6 neurons at second hidden layer,etc
run:
	./array_ser_main 5 6

runP:
	./array_par_main 5 6

log:
	./array_ser_main > log

clean:
	rm array_ser_main.o array_ser_main array_par_main array_par_main.o log

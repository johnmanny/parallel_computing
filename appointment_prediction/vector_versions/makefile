# Authors: John Nemeth
COMPILER= g++
CFLAGS= -g -Wall -std=c++11 -m64 -O3


ser_main: ser_main.o
	$(COMPILER) $(CFLAGS) ser_main.o -o ser_main

ser_main.o: ser_main.cpp
	$(COMPILER) $(CFLAGS) -c ser_main.cpp -o ser_main.o

par_main: par_main.o
	$(COMPILER) $(CFLAGS) -fopenmp par_main.o -o par_main

par_main.o: par_main.cpp
	$(COMPILER) $(CFLAGS) -fopenmp -c par_main.cpp -o par_main.o

# Arguments are number of hidden neurons on each layer
#   I.E.: 5 neurons at first hidden layer, 6 neurons at second hidden layer,etc
run:
	./ser_main 7 5 3

runP:
	./par_main 7 5 3

log:
	./ser_main > log

clean:
	rm ser_main.o ser_main par_main par_main.o log batch_main

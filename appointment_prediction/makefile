# Authors: John Nemeth
COMPILER= g++
CFLAGS= -g -Wall -std=c++11 


ser_main: ser_main.o
	$(COMPILER) $(CFLAGS) ser_main.o -o ser_main

ser_main.o: ser_main.cpp
	$(COMPILER) $(CFLAGS) -c ser_main.cpp -o ser_main.o

# Arguments are number of hidden neurons on each layer
#   I.E.: 5 neurons at first hidden layer, 6 neurons at second hidden layer,etc
run:
	./ser_main 5 6 3
log:
	./ser_main > log

clean:
	rm ser_main.o ser_main log

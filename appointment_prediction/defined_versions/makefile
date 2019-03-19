# Authors: John Nemeth
COMPILER= g++
CFLAGS= -g -Wall -std=c++11 


oser_main: oser_main.o
	$(COMPILER) $(CFLAGS) oser_main.o -o oser_main

oser_main.o: oser_main.cpp
	$(COMPILER) $(CFLAGS) -c oser_main.cpp -o oser_main.o

run:
	./oser_main
log:
	./oser_main > log

clean:
	rm oser_main.o oser_main log

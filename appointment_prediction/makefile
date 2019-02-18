# Authors: John Nemeth
COMPILER= g++
CFLAGS= -g -Wall -std=c++11 

all: ser_main

ser_main.o: ser_main.cpp
	$(COMPILER) $(CFLAGS) -c ser_main.cpp -o ser_main.o

ser_main: ser_main.o
	$(COMPILER) $(CFLAGS) ser_main.o -o ser_main

clean:
	rm ser_main.o ser_main

# author: john nemeth

all:
	g++ -I/opt/X11/include -g -o gb main.cpp -L/opt/X11/lib -lX11 -ljpeg -lpthread

gs:
	g++ -I/opt/X11/include -g -o gs grayscale.cpp -L/opt/X11/lib -lX11 -ljpeg -lpthread

clean:
	rm gb gs

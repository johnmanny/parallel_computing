# author: john nemeth

all:
	g++ -I/opt/X11/include -g -o blur_serial blur_serial.cpp -L/opt/X11/lib -lX11 -ljpeg -lpthread -std=c++11

ispc:
	g++ -I/opt/X11/include -g -o blur_ispc blur_ispc.cpp -L/opt/X11/lib -lX11 -ljpeg -lpthread -std=c++11

gs:
	g++ -I/opt/X11/include -g -o gs grayscale.cpp -L/opt/X11/lib -lX11 -ljpeg -lpthread 

clean:
	rm blur_serial gs

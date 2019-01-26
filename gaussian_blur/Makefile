# author: john nemeth

serial:
	g++ -I/opt/X11/include -g -o blur_serial blur_serial.cpp -L/opt/X11/lib -lX11 -ljpeg -lpthread -std=c++11

par: ispc
	g++ -I/opt/X11/include -g -o blur_ispc blur_ispc.cpp objs/vecblur_ispc.o -L/opt/X11/lib -lX11 -ljpeg -lpthread -std=c++11

ispc:
	ispc -Iobjs/ -Wall vecblur.ispc -o objs/vecblur_ispc.o -h vecblur_ispc.h

gs:
	g++ -I/opt/X11/include -g -o gs grayscale.cpp -L/opt/X11/lib -lX11 -ljpeg -lpthread 

clean:
	rm blur_serial gs

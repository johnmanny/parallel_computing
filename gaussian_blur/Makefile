# author: john nemeth

serial:
	g++ -I/opt/X11/include -g -o blur_serial blur_serial.cpp -L/opt/X11/lib -lX11 -ljpeg -lpthread -std=c++11

par: ispc
	g++ -m64 -I/opt/X11/include -g -o blur_ispc blur_ispc.cpp vecblur_ispc.o -L/opt/X11/lib -lX11 -ljpeg -lpthread -std=c++11

ispc:
	ispc -Iobjs -O2 --arch=x86-64 --target=sse2 vecblur.ispc -o vecblur_ispc.o -h vecblur_ispc.h

gs:
	g++ -I/opt/X11/include -g -o gs grayscale.cpp -L/opt/X11/lib -lX11 -ljpeg -lpthread 
runs:
	./blur_serial
runp:
	./blur_ispc
clean:
	rm blur_serial gs output.jpg blur_ispc vecblur_ispc.h vecblur_ispc.o

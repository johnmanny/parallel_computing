/*
    Author: John Nemeth
    Sources: class/assignment files
    Description: a project which implements gaussian blur serially
*/

#include <cstdio>
#include <iostream>

#define cimg_use_jpeg
#include "CImg.h"
using namespace cimg_library;

int main () {

    // import image from jpeg file
    CImg<unsigned char> inputImg("bombus.jpg");

    // define height values
    int width = inputImg.width();
    int height = inputImg.height();

    // create arrays for pixel RGB's
    int red[width * height];
    int green[width * height];
    int blue[width * height];

    cimg_forXY(inputImg, x, y) {
        int index = (x * width) + y;
        red[index] = (int)inputImg(x, y, 0, 0);
        green[index] = (int)inputImg(x, y, 0, 1);
        blue[index] = (int)inputImg(x, y, 0, 2);
    }


    return 0;
}

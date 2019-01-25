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

    // extract dimensions
    int width = (int)inputImg.width();
    int height = (int)inputImg.height();
    int pixels = width * height;
    // create output image
    CImg<unsigned char> outputImg(inputImg.width(), inputImg.height(), 1, 3);

    /*
    // create arrays for pixel RGB's
    float red[width * height];
    float green[width * height];
    float blue[width * height];
    int colStart = 0;
    */
    //printf("height: %d\nwidth: %d\n", height, width);

    // create array for red
    float ** R = new float*[width];
    float ** G = new float*[width];
    float ** B = new float*[width];
    for (int i = 0; i < width; i++) {
        R[i] = new float[height];
        G[i] = new float[height];
        B[i] = new float[height];
    }
    /*
    float * R = new float[pixels];
    float * G = new float[pixels];
    float * B = new float[pixels];
    int colStart = 0;

    // grab RBG's of input image
    for (int c = 0; c < width; c++) {
        colStart = c * height;
        for (int r = 0; r < height; r++) {
            R[colStart + r] = inputImg(c, r, 0);
            G[colStart + r] = inputImg(c, r, 1);
            B[colStart + r] = inputImg(c, r, 2);
        }
    }
    */

    // grab pixels values
    for (int c = 0; c < width; c++) {
        for (int r = 0; r < height; r++) {
            R[c][r] = inputImg(c, r, 0);
            G[c][r] = inputImg(c, r, 1);
            B[c][r] = inputImg(c, r, 2);
        }
    }

    // create array for red
    float ** Rmod = new float*[width];
    float ** Gmod = new float*[width];
    float ** Bmod = new float*[width];
    for (int i = 0; i < width; i++) {
        Rmod[i] = new float[height];
        Gmod[i] = new float[height];
        Bmod[i] = new float[height];
    }

    // init
    for (int c = 0; c < width; c++) {
        for (int r = 0; r < height; r++) {
            Rmod[c][r] = R[c][r];
            Gmod[c][r] = G[c][r];
            Bmod[c][r] = B[c][r];
        }
    }

    // compute averaged values
    for (int c = 0; c < width; c++) {
        for (int r = 0; r < height; r++) {
            if ((c == 0) || (r == 0) || (c == (width-1)) || (r == (height - 1)))
                continue;
            Rmod[c][r] += R[c-1][r-1] + R[c][r-1] + R[c-1][r+1] + R[c-1][r] + R[c+1][r] + R[c-1][r+1] + R[c][r+1] + R[c+1][r+1];
            Gmod[c][r] += G[c-1][r-1] + G[c][r-1] + G[c-1][r+1] + G[c-1][r] + G[c+1][r] + G[c-1][r+1] + G[c][r+1] + G[c+1][r+1];
            Bmod[c][r] += B[c-1][r-1] + B[c][r-1] + B[c-1][r+1] + B[c-1][r] + B[c+1][r] + B[c-1][r+1] + B[c][r+1] + B[c+1][r+1];
            //Rmod[c][r] += ((c % width != 0) && (r % height != 0)) ? (R[c-1][r-1] + R[c][r-1] + R[c-1][r+1] + R[c-1][r] + R[c+1][r] + R[c-1][r+1] + R[c][r+1] + R[c+1][r+1]) : 0.0;
           
        }
    }
    /*
    // compute average of RGB
    //     index edge cases: 0 to height, 2height-1 to 2height...
    //                       (width-2)*height-1 to (width-2)*height, (width-1)height to width*height-1
    for (int i = 0; i < pixels; i++) {
        if ((i % width == 0) || (i % height == 0)) {

        }

    }
    */
    /*
    // compute vertical edge cases
    for (int c = 0; c < 2; c++) {
        for (int r = 0; r < width; r++) {

        }
    }
    */
    printf("height: %d\nwidth: %d\nR[0][5] val: %f\n", height, width, R[0][5]);
    // delete dynamic cols
    for (int i = 0; i < width; i++) {
        delete [] R[i];
        delete [] G[i];
        delete [] B[i];
    }
    delete [] R;
    delete [] G;
    delete [] B;
    /*
    // iterate over vertical edges of image
    //  8 neighbors of vertical edge case
    for (int c = 0; c < 2; c++) {
        for (int r = 0; r < height; r++) {
            
        }
    }
    */


    //std::cout << "red 5" << red[5] << endl;
    outputImg = inputImg;
    outputImg.save_jpeg("output.jpg");
    //return 0;
}

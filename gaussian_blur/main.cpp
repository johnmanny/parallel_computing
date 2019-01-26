/*
    Author: John Nemeth
    Sources: CImg documentation
    Description: a project which implements gaussian blur serially. adds border rows
		and columns for easy handling of image edge cases
*/

#include <cstdio>
#include <iostream>
#include <sys/time.h>

#define cimg_use_jpeg
#include "CImg.h"
using namespace cimg_library;

int main () {

    // declare loop variables
    int c, r, i;

    // import image from jpeg file
    CImg<unsigned char> inputImg("bombus.jpg");
    printf("\tinput image 'bombus.jpg'\n");

    // extract dimensions + add 'border' for picture (for zero values - fixes edge cases)
    int width = (int)inputImg.width() + 2;
    int height = (int)inputImg.height() + 2;
    int realWidth = width - 2;
    int realHeight = height - 2;

    // create output image
    CImg<unsigned char> outputImg(inputImg.width(), inputImg.height(), 1, 3);

    // create arrays for manipulations, adding surrounding 'borders'
    float ** R = new float*[width];
    float ** G = new float*[width];
    float ** B = new float*[width];
    for (i = 0; i < width; i++) {
        R[i] = new float[height];
        G[i] = new float[height];
        B[i] = new float[height];
    }

    // initialize all positions to zero (including border rows and columns)
    for (c = 0; c < width; c++) {
        for (r = 0; r < height; r++) {
            R[c][r] = 0.0;
            G[c][r] = 0.0;
            B[c][r] = 0.0;
        }
    }

    // grab pixels values for image and place inside border rows and columns
    for (c = 1; c < (width - 1); c++) {
        int imageX = c - 1;
        for (r = 1; r < (height - 1); r++) {
            int imageY = r - 1;
            R[c][r] = inputImg(imageX, imageY, 0);
            G[c][r] = inputImg(imageX, imageY, 1);
            B[c][r] = inputImg(imageX, imageY, 2);
        }
    }

    // create array for modified image
    float ** Rmod = new float*[realWidth];
    float ** Gmod = new float*[realWidth];
    float ** Bmod = new float*[realWidth];
    for (i = 0; i < realWidth; i++) {
        Rmod[i] = new float[realHeight];
        Gmod[i] = new float[realHeight];
        Bmod[i] = new float[realHeight];
    }

    // initialize all values of modified image to 0
    for (c = 0; c < realWidth; c++) {
        for (r = 0; r < realHeight; r++) {
            Rmod[c][r] = 0.0;
            Gmod[c][r] = 0.0;
            Bmod[c][r] = 0.0;
        }
    }

    // define timer
    struct timeval end, start;
    gettimeofday(&start, NULL); 

    //// compute averaged values
    //		top left = tX-1,tY-1 --- top = tX, tY-1 --- top right = tX+1, tY-1
    //		left = tx-1, ty --- right = tX+1, tY
    // 		bot left = tX-1, tY+1 --- bot = tX, tY+1 --- bot right = tX+1, tY+1
    int tX, tY;
    for (c = 0; c < realWidth; c++) {
        tX = c + 1;		// transformed x
        for (r = 0; r < realHeight; r++) {
            tY = r + 1;		// transformed y
            Rmod[c][r] = (R[tX][tY] + R[tX-1][tY-1] + R[tX][tY-1] + R[tX+1][tY-1] + R[tX-1][tY] + R[tX+1][tY] + R[tX-1][tY+1] + R[tX][tY+1] + R[tX+1][tY+1]) / 9;
            Gmod[c][r] = (G[tX][tY] + G[tX-1][tY-1] + G[tX][tY-1] + G[tX+1][tY-1] + G[tX-1][tY] + G[tX+1][tY] + G[tX-1][tY+1] + G[tX][tY+1] + G[tX+1][tY+1]) / 9;
            Bmod[c][r] = (B[tX][tY] + B[tX-1][tY-1] + B[tX][tY-1] + B[tX+1][tY-1] + B[tX-1][tY] + B[tX+1][tY] + B[tX-1][tY+1] + B[tX][tY+1] + B[tX+1][tY+1]) / 9;
        }
    }

    // get end time, calculate time passed and output
    gettimeofday(&end, NULL);

    long long tPassed = (end.tv_sec - start.tv_sec) * 1000000LL + (end.tv_usec - start.tv_usec);
    printf("\tserial implementation calculation time: %f seconds\n", tPassed/100000.0);

    // set output image to modified array values (blur values)
    for (c = 0; c < realWidth; c++) {
        for (r = 0; r < realHeight; r++) {
            outputImg(c, r, 0) = Rmod[c][r];
            outputImg(c, r, 1) = Gmod[c][r];
            outputImg(c, r, 2) = Bmod[c][r];
        }
    }

    // save to output image
    outputImg.save_jpeg("output.jpg");
    printf("\timage output to 'output.jpg'\n");

    // de-allocate original image arrays
    for (i = 0; i < width; i++) {
        delete [] R[i];
        delete [] G[i];
        delete [] B[i];
    }
    delete [] R;
    delete [] G;
    delete [] B;

    // de-allocate modified image arrays
    for (i = 0; i < realWidth; i++) {
        delete [] Rmod[i];
        delete [] Gmod[i];
        delete [] Bmod[i];
    }
    delete [] Rmod;
    delete [] Gmod;
    delete [] Bmod;

    return 0;
}

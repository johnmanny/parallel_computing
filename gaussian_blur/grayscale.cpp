// OSX compilation:
//    g++ -I/opt/X11/include -o grayscale grayscale.cpp -L/opt/X11/lib -lX11 -ljpeg

#include <cstdio>
#include <iostream>

#define cimg_use_jpeg
#include "CImg.h"

int main() {

  // import image from jpg file
  cimg_library::CImg<unsigned char> input_img("bombus.jpg");

  // create new image (width, height, depth, channels (RGB))
  cimg_library::CImg<unsigned char> output_img(
      input_img.width(), input_img.height(), 1, 3);

  // iterate over the input image
  for (int c = 0; c < input_img.width(); ++c) {
    for (int r = 0; r < input_img.height(); ++r) {

      // extract RGB values
      float R = input_img(c, r, 0);
      float G = input_img(c, r, 1);
      float B = input_img(c, r, 2);

      // compute the average (grayscale)
      float avg = (R + G + B) / 3;

      // write to the output image
      output_img(c, r, 0) = output_img(c, r, 1) = output_img(c, r, 2) = avg; 

    }
  }
      
  output_img.save_jpeg("output.jpg");
}

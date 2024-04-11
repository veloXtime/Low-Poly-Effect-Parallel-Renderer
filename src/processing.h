#ifndef PROCESSING_H
#define PROCESSING_H
#include "CImg.h"

unsigned char* gaussianBlur(const unsigned char* inputImage, int width,
                            int height, int channels);
cimg_library::CImg<unsigned char> extractEdge(
    cimg_library::CImg<unsigned char>& image);
#endif

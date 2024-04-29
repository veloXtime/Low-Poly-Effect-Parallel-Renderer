#ifndef PROCESSING_H
#define PROCESSING_H
#include "CImg.h"
unsigned char* gaussianBlurCPU(const unsigned char* inputImage, int width,
                               int height, int channels);
unsigned char* gaussianBlur(const unsigned char* inputImage, int width,
                            int height, int channels);
cimg_library::CImg<unsigned char> extractEdge(
    cimg_library::CImg<unsigned char>& image);

cimg_library::CImg<unsigned char> extractEdgeCanny(
    cimg_library::CImg<unsigned char>& image, int method = 0);

cimg_library::CImg<unsigned char> edgeDraw(
    cimg_library::CImg<unsigned char>& image, int method = 0);
#endif
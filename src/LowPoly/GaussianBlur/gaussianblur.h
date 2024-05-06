#ifndef GAUSSIAN_BLUR_H
#define GAUSSIAN_BLUR_H

#include "CImg.h"

using CImg = cimg_library::CImg<unsigned char>;
using CImgBool = cimg_library::CImg<bool>;
using CImgFloat = cimg_library::CImg<float>;

const int THREAD_PER_BLK = 256;
// const double BLUR_SIGMA = 1;
const double BLUR_SIGMA = 2.3;
// const int BLUR_RADIUS = 2;
const int BLUR_RADIUS = 7;
const int BLUR_WIDTH = 2 * BLUR_RADIUS + 1;

// Creates a Gaussian kernel with input radius
unsigned char *gaussianBlurCPU(const unsigned char *inputImage, int width,
                               int height, int channels);
unsigned char *gaussianBlur(const unsigned char *inputImage, int width,
                            int height, int channels);

void gpuWarmUp();

#endif
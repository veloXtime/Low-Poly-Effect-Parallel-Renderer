#ifndef GAUSSIAN_BLUR_H
#define GAUSSIAN_BLUR_H


const int THREAD_PER_BLK = 256;
const double BLUR_SIGMA = 1;
const int BLUR_RADIUS = 2;
const int BLUR_WIDTH = 2 * BLUR_RADIUS + 1;

// Creates a Gaussian kernel with input radius
double* gaussianKernel();

__global__ void rgb_to_grayscale_kernel(const unsigned char* inputImage,
                                        unsigned char* grayscaleImage,
                                        int width, int height);

__constant__ double
    kernelConstant[(2 * BLUR_RADIUS + 1) * (2 * BLUR_RADIUS + 1)];
#endif
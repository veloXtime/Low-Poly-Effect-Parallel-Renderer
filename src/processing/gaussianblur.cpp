#include <math.h>

#include <algorithm>
#include <iostream>

#include "processing.h"
#include "gaussianblur.h"

/**
 * Create and return a Gaussian kernel for convolution
 * @param radius kernel radius
 * @param sigma  Gaussian kernel standard deviation
 */
double* gaussianKernel(const int radius, const int sigma) {
    const int width = radius * 2 + 1;

    double * kernel = (double *)malloc(sizeof(double) * width * width);
    double sum = 0.0;
    // Populate every position in the kernel with the respective Gaussian
    // distribution value
    std::cout << "\n kernel: \n";
    for (int x = -radius; x <= radius; x++) {
        for (int y = -radius; y <= radius; y++) {
            double expNumerator = -(x * x + y * y);
            double expDenominator = 2.0 * sigma * sigma;
            double eExpression = exp(expNumerator / expDenominator);
            double kernelValue =
                eExpression / (2.0 * M_PI * sigma * sigma);
            size_t index = (x + radius) * width + y + radius;
            kernel[index] = kernelValue;
            sum += kernelValue;
        }
    }

    // Normalize the kernel
    for (int i = 0; i < width * width; i++) {
        kernel[i] /= sum;
    }

    return kernel;
}

unsigned char* gaussianBlur(const unsigned char* inputImage, int width,
                            int height, int channels) {
    // Copy kernel to constant memory
    double* kernel = gaussianKernel(BLUR_RADIUS, BLUR_SIGMA);
    cudaMemcpyToSymbol(kernelConstant, kernel,
                       sizeof(double) * BLUR_WIDTH * BLUR_WIDTH);

    // Total image size each pixel 3 chars for R,G,B
    int imageDataSize = sizeof(unsigned char) * width * height * channels;

    // Allocate memory for output blurred image
    unsigned char* outputImage = (unsigned char*)malloc(imageDataSize);

    // Allocate device memory
    unsigned char* inputImageDevice, *outputImageDevice;
    cudaMalloc(&inputImageDevice, imageDataSize);
    cudaMemcpy(inputImageDevice, inputImage, imageDataSize,
               cudaMemcpyHostToDevice);
    cudaMalloc(&outputImageDevice, imageDataSize);

  

    // CUDA memory allocation and kernel invocation
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    gaussian_blur_kernel<<<gridSize, blockSize>>>(
        inputImageDevice, outputImageDevice, width, height, channels);

    // Copy the result back to host
    cudaMemcpy(outputImage, outputImageDevice, imageDataSize,
               cudaMemcpyDeviceToHost);

    // unsigned char* grayscaleImage;
    // cudaMallocManaged(&grayscaleImage, sizeof(unsigned char) * width * height);

    // rgb_to_grayscale_kernel<<<gridSize, blockSize>>>(outputImageDevice, grayscaleImage, width, height);

    cudaDeviceSynchronize();

    // Free device memory
    cudaFree(inputImageDevice);
    cudaFree(outputImageDevice);

    return outputImage;
}

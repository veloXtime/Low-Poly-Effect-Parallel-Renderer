#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <math.h>

#include <algorithm>
#include <iostream>

#include "processing.h"

const int THREAD_PER_BLK = 256;
const double BLUR_SIGMA = 1.5;
const int BLUR_RADIUS = 5;

// The constant Gaussian kernel that reside in Global GPU memory
__constant__ double
    kernelConstant[(2 * BLUR_RADIUS + 1) * (2 * BLUR_RADIUS + 1)];

/**
 * CUDA Gaussian blur function
 * @param inputImage input image ready for processing
 * @param outputImage output blurred image
 * @param width image width
 * @param height image height
 * @param kernelWidth kernel width
 * @param channels image channels, 3 if RGB, 1 if greyscale
 */
__global__ void gaussian_blur_kernel(const unsigned char* inputImage,
                                     unsigned char* outputImage, int width,
                                     int height, int kernelWidth,
                                     int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (row * width + col) * channels;

    if (col < width && row < height) {
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;

            for (int i = -BLUR_RADIUS; i <= BLUR_RADIUS; ++i) {
                for (int j = -BLUR_RADIUS; j <= BLUR_RADIUS; ++j) {
                    int newRow = min(max(row + i, 0), height - 1);
                    int newCol = min(max(col + j, 0), width - 1);
                    int newIdx = c * width * height + newRow * width + newCol;

                    float weight =
                        kernelConstant[(i + BLUR_RADIUS) * kernelWidth +
                                       (j + BLUR_RADIUS)];
                    sum += double(inputImage[newIdx]) * weight;
                }
            }

            outputImage[c * width * height + row * width + col] =
                static_cast<unsigned char>(sum);
        }
    }
}

/**
 * Creates a Gaussian blur kernel for convolution
 * @param radius radius of the kernel
 * @param sigma  standard deviation of the kernel
 */
double* gaussianKernel(int radius, int sigma) {
    int kernelWidth = 2 * radius + 1;
    double kernel[kernelWidth * kernelWidth] =
        (double*)malloc(sizeof(double) * kernelWidth * kernelWidth);
    double sum = 0.0;
    // Populate every position in the kernel with the respective Gaussian
    // distribution value
    for (int x = -radius; x <= radius; x++) {
        for (int y = -radius; y <= radius; y++) {
            double expNumerator = -(x * x + y * y);
            double expDenominator = 2.0 * sigma * sigma;
            double eExpression = exp(expNumerator / expDenominator);
            double kernelValue = eExpression / (2.0 * M_PI * sigma * sigma);
            size_t index = (x + radius) * kernelWidth + y + radius;
            kernel[index] = kernelValue;
            sum += kernelValue;
        }
    }

    // Normalize the kernel
    for (int i = 0; i < kernelWidth * kernelWidth; i++) {
        kernel[i] /= sum;
    }

    return kernel;
}

/**
 * Gaussian blur function that takes an input image and returns blurred image
 * @param inputImage input image
 * @param width width of input image
 * @param height height of input image
 * @param channels number of color channels input image has
 */
unsigned char* gaussianBlur(const unsigned char* inputImage, int width,
                            int height, int channels) {
    int kernelWidth = 2 * BLUR_RADIUS + 1;
    double* kernel = gaussianKernel(BLUR_RADIUS, BLUR_SIGMA);
    double sum = 0.0;
    // Populate every position in the kernel with the respective Gaussian
    // distribution value
    for (int x = -BLUR_RADIUS; x <= BLUR_RADIUS; x++) {
        for (int y = -BLUR_RADIUS; y <= BLUR_RADIUS; y++) {
            double expNumerator = -(x * x + y * y);
            double expDenominator = 2.0 * BLUR_SIGMA * BLUR_SIGMA;
            double eExpression = exp(expNumerator / expDenominator);
            double kernelValue =
                eExpression / (2.0 * M_PI * BLUR_SIGMA * BLUR_SIGMA);
            size_t index = (x + BLUR_RADIUS) * kernelWidth + y + BLUR_RADIUS;
            kernel[index] = kernelValue;
            sum += kernelValue;
        }
    }

    // Normalize the kernel
    for (int i = 0; i < kernelWidth * kernelWidth; i++) {
        kernel[i] /= sum;
    }

    // Copy kernel to constant memory
    cudaMemcpyToSymbol(kernelConstant, kernel,
                       sizeof(double) * kernelWidth * kernelWidth);

    unsigned char* outputImage = (unsigned char*)malloc(
        sizeof(unsigned char) * width * height * channels);

    // Allocate device memory
    int imageDataSize = sizeof(unsigned char) * width * height * channels;
    unsigned char* inputImageDevice;
    unsigned char* outputImageDevice;
    cudaMalloc(&inputImageDevice, imageDataSize);
    cudaMemcpy(inputImageDevice, inputImage, imageDataSize,
               cudaMemcpyHostToDevice);
    cudaMalloc(&outputImageDevice, imageDataSize);

    // CUDA memory allocation and kernel invocation
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    gaussian_blur_kernel<<<gridSize, blockSize>>>(
        inputImageDevice, outputImageDevice, width, height, kernelWidth,
        channels);

    // Copy the result back to host
    cudaMemcpy(outputImage, outputImageDevice, imageDataSize,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(inputImageDevice);
    cudaFree(outputImageDevice);

    return outputImage;
}

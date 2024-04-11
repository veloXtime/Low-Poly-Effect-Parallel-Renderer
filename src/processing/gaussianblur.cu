#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <math.h>

#include <algorithm>
#include <iostream>

#include "processing.h"

const int THREAD_PER_BLK = 256;
const double GAUSSIAN_BLUR_SIGMA = 1.5;
const int GAUSSIAN_BLUR_RADIUS = 5;

__constant__ double kernelConstant[(2 * GAUSSIAN_BLUR_RADIUS + 1) *
                                   (2 * GAUSSIAN_BLUR_RADIUS + 1)];

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

            for (int i = -GAUSSIAN_BLUR_RADIUS; i <= GAUSSIAN_BLUR_RADIUS;
                 ++i) {
                for (int j = -GAUSSIAN_BLUR_RADIUS; j <= GAUSSIAN_BLUR_RADIUS;
                     ++j) {
                    int newRow = min(max(row + i, 0), height - 1);
                    int newCol = min(max(col + j, 0), width - 1);
                    int newIdx = (newRow * width + newCol) * channels + c;

                    float weight = kernelConstant[(i + GAUSSIAN_BLUR_RADIUS) *
                                                      kernelWidth +
                                                  (j + GAUSSIAN_BLUR_RADIUS)];
                    sum += double(inputImage[newIdx]) * weight;
                }
            }

            outputImage[idx + c] = static_cast<unsigned char>(sum);
        }
    }
}

unsigned char* gaussianBlur(const unsigned char* inputImage, int width,
                            int height, int channels) {
    int kernelWidth = 2 * GAUSSIAN_BLUR_RADIUS + 1;
    double kernel[kernelWidth * kernelWidth];
    double sum = 0.0;
    // Populate every position in the kernel with the respective Gaussian
    // distribution value
    for (int x = -GAUSSIAN_BLUR_RADIUS; x <= GAUSSIAN_BLUR_RADIUS; x++) {
        for (int y = -GAUSSIAN_BLUR_RADIUS; y <= GAUSSIAN_BLUR_RADIUS; y++) {
            double expNumerator = -(x * x + y * y);
            double expDenominator =
                2.0 * GAUSSIAN_BLUR_SIGMA * GAUSSIAN_BLUR_SIGMA;
            double eExpression = exp(expNumerator / expDenominator);
            double kernelValue =
                eExpression /
                (2.0 * M_PI * GAUSSIAN_BLUR_SIGMA * GAUSSIAN_BLUR_SIGMA);
            size_t index = (x + GAUSSIAN_BLUR_RADIUS) * kernelWidth + y +
                           GAUSSIAN_BLUR_RADIUS;
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

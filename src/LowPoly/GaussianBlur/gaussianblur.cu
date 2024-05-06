#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <math.h>

#include <algorithm>
#include <iostream>

#include "gaussianblur.h"

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
    double* kernel =
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
 * Gaussian blur function that takes an input image and returns blurred
 * image
 * @param inputImage input image
 * @param width width of input image
 * @param height height of input image
 * @param channels number of color channels input image has
 */
unsigned char* gaussianBlur(const unsigned char* inputImage, int width,
                            int height, int channels) {
    // Create Gaussian blur kernel
    int kernelWidth = 2 * BLUR_RADIUS + 1;
    double* kernel = gaussianKernel(BLUR_RADIUS, BLUR_SIGMA);
    // Copy kernel to constant memory
    cudaMemcpyToSymbol(kernelConstant, kernel,
                       sizeof(double) * kernelWidth * kernelWidth);

    // Allocate memory for output image
    int imageDataSize = sizeof(unsigned char) * width * height * channels;
    unsigned char* outputImage = (unsigned char*)malloc(imageDataSize);

    // Allocate device memory
    unsigned char* inputImageDevice;
    unsigned char* outputImageDevice;
    cudaMalloc(&inputImageDevice, imageDataSize);
    cudaMemcpy(inputImageDevice, inputImage, imageDataSize,
               cudaMemcpyHostToDevice);
    cudaMalloc(&outputImageDevice, imageDataSize);

    // CUDA memory allocation
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Kernel invocation
    gaussian_blur_kernel<<<gridSize, blockSize>>>(
        inputImageDevice, outputImageDevice, width, height, kernelWidth,
        channels);

    // Copy the result back to host
    cudaMemcpy(outputImage, outputImageDevice, imageDataSize,
               cudaMemcpyDeviceToHost);

    // Free device memory and kernel
    cudaFree(inputImageDevice);
    cudaFree(outputImageDevice);
    free(kernel);

    return outputImage;
}

/**
 * CPU version of Gaussian blur, used as a baseline for testing performance
 * @param inputImage input image
 * @param width width of input image
 * @param height height of input image
 * @param channels number of color channels input image has
 */
unsigned char* gaussianBlurCPU(const unsigned char* inputImage, int width,
                               int height, int channels) {
    // Create Gaussian kernel
    int kernelWidth = 2 * BLUR_RADIUS + 1;
    double* kernel = gaussianKernel(BLUR_RADIUS, BLUR_SIGMA);
    unsigned char* outputImage = (unsigned char*)malloc(
        sizeof(unsigned char) * width * height * channels);

    // Convolve over the input image
    // for (int i = 0; i < 300; i++) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            for (int c = 0; c < channels; ++c) {
                double sum = 0.0f;

                for (int i = -BLUR_RADIUS; i <= BLUR_RADIUS; ++i) {
                    for (int j = -BLUR_RADIUS; j <= BLUR_RADIUS; ++j) {
                        int newRow = min(max(row + i, 0), height - 1);
                        int newCol = min(max(col + j, 0), width - 1);
                        int newIdx =
                            c * width * height + newRow * width + newCol;

                        double weight = kernel[(i + BLUR_RADIUS) * kernelWidth +
                                               (j + BLUR_RADIUS)];
                        sum += double(inputImage[newIdx]) * weight;
                    }
                }

                outputImage[c * width * height + row * width + col] =
                    static_cast<unsigned char>(sum);
            }
        }
    }
    // }

    // Free Gaussian kernel
    free(kernel);

    return outputImage;
}

__global__ void warmupKernel() {};

void gpuWarmUp() {
    // Launch the warm-up kernel with a single thread
    warmupKernel<<<1, 1>>>();

    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    // Wait for the GPU to finish
    cudaDeviceSynchronize();
}
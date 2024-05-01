#include <cuda_runtime.h>

#include "edgedraw.h"

__constant__ int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
__constant__ int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

__global__ void colorToGrayKernel(unsigned char *image,
                                  unsigned char *grayImage, int width,
                                  int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;  // 3 channels per pixel
        unsigned char r = image[idx];
        unsigned char g = image[idx + 1];
        unsigned char b = image[idx + 2];
        unsigned char grayValue = 0.299f * r + 0.587f * g + 0.114f * b;
        grayImage[y * width + x] = grayValue;
    }
}

__global__ void gradientCalculationKernel(unsigned char *grayImage,
                                          unsigned char *gradient,
                                          float *direction, int width,
                                          int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        float gx = 0.0f, gy = 0.0f;

        // Apply the Sobel filter
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int pixel = grayImage[(y + j) * width + (x + i)];
                gx += sobel_x[i + 1][j + 1] * pixel;
                gy += sobel_y[i + 1][j + 1] * pixel;
            }
        }

        int idx = y * width + x;
        gradient[idx] = sqrtf(gx * gx + gy * gy);
        direction[idx] = atan2f(gy, gx);
    }
}

void gradientInGrayGPU(CImg &image, CImg &gradient, CImgFloat &direction) {
    int width = image.width(), height = image.height();

    // Flatten the image data for CUDA
    unsigned char *d_image, *d_grayImage, *d_gradient;
    float *d_direction;
    size_t imageSize = width * height * 3 * sizeof(unsigned char);
    size_t grayImageSize = width * height * sizeof(unsigned char);
    size_t directionSize = width * height * sizeof(float);

    cudaMalloc(&d_image, imageSize);
    cudaMalloc(&d_grayImage, grayImageSize);
    cudaMalloc(&d_gradient, grayImageSize);
    cudaMalloc(&d_direction, directionSize);

    cudaMemcpy(d_image, image.data(), imageSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    colorToGrayKernel<<<gridSize, blockSize>>>(d_image, d_grayImage, width,
                                               height);
    gradientCalculationKernel<<<gridSize, blockSize>>>(
        d_grayImage, d_gradient, d_direction, width, height);

    // Copy results back to host
    cudaMemcpy(gradient.data(), d_gradient, grayImageSize,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(direction.data(), d_direction, directionSize,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_image);
    cudaFree(d_grayImage);
    cudaFree(d_gradient);
    cudaFree(d_direction);
}

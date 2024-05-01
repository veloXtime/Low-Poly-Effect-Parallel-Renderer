#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

/**
 * Convert a rgb image to greyscale.
 *
 * @param inputImage RGB image in CUDA memory.
 * @param greyscaleImage output greyscale image in CUDA memory.
 * @param width image width.
 * @param height image height.
 */
__global__ void rgb_to_grayscale_kernel(const unsigned char* inputImage,
                                        unsigned char* grayscaleImage,
                                        int width, int height) {
    // Calculate our pixel's location
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        // Get the index in the array for the pixel
        int rgb_idx = row * width + col;
        int grayscale_idx = rgb_idx * 3;  // 3 channels

        // Extract the R, G, B values
        unsigned char r = inputImage[grayscale_idx + 0];
        unsigned char g = inputImage[grayscale_idx + 1];
        unsigned char b = inputImage[grayscale_idx + 2];

        // Perform the conversion to grayscale
        unsigned char gray =
            static_cast<unsigned char>(int(0.299f * r + 0.587f * g + 0.114f * b));

        // Write the grayscale intensity to the output image
        grayscaleImage[rgb_idx] = gray;
    }
}

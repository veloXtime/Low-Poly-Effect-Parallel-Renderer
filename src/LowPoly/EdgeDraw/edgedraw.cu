#include <cuda_runtime.h>

#include "edgedraw.h"

CImg edgeDrawGPU(const CImg &image, int method) {
    // Create a new image to store the edge
    CImg gradient(image.width(), image.height());
    CImgFloat direction(image.width(), image.height());

    // Calculate gradient magnitude for each pixel
    if (method == 0) {
        gradientInGrayGPU(image, gradient, direction);
    } else {
        gradientInColor(image, gradient, direction);
    }
}

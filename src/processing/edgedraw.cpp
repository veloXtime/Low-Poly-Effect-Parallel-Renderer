#include <iostream>

#include "CImg.h"

using CImg = cimg_library::CImg<unsigned char>;

/**
 * Extract edges from the image using Canny edge detection method.
 *
 * @param image Image to extract edge, with RGB color and single depth.
 * @pre Noise should have been removed on the image in previous steps.
 */
CImg extractEdgeCanny(CImg &image) {
    // Create a new image to store the edge
    CImg edge(image.width(), image.height(), 1, 3, 0);

    // Calculate gradient magnitude for each pixel
    cimg_forXY(image, x, y) {
        // If the pixel is not at the edge of the image
        if (x > 0 && x < image.width() - 1 && y > 0 && y < image.height() - 1) {
            // Calculate the gradient in the x and y directions
            int gradientX = image(x + 1, y) - image(x - 1, y);
            int gradientY = image(x, y + 1) - image(x, y - 1);

            // Calculate the magnitude of the gradient
            int magnitude = sqrt(gradientX * gradientX + gradientY * gradientY);

            // Set the pixel value in the edge image
            edge(x, y) = magnitude;
        }
    }

    return edge;
}

/**
 * Convert colored image to grayscale and calculate gradient
 */
int gradientInGray(CImg &image, CImg &edge) {}

/**
 * Calculate gradient separately in RGB dimension and combine
 */
int gradientInColor(CImg &image, CImg &edge) {}
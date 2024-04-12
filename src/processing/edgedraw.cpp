#include <iostream>

#include "CImg.h"

using CImg = cimg_library::CImg<unsigned char>;

/**
 * Extract edges from the image using Canny edge detection method.
 *
 * @param image Image to extract edge, with RGB color and single depth.
 * @param method Method to use for edge detection, 0 for grayscale, 1 for RGB.
 * @pre Noise should have been removed on the image in previous steps.
 */
CImg extractEdgeCanny(CImg &image, int method) {
    // Create a new image to store the edge
    int **gradient = new int *[image.width()];
    for (int i = 0; i < image.width(); i++) {
        gradient[i] = new int[image.height()];
    }

    // Calculate gradient magnitude for each pixel
    if (method == 0) {
        gradientInGray(image, gradient);
    } else {
        gradientInColor(image, gradient);
    }

    return edge;
}

/**
 * Convert colored image to grayscale and calculate gradient
 */
void gradientInGray(CImg &image, CImg &gradient) {
    // Convert the image to grayscale
    CImg<unsigned char> grayImage(image.width(), image.height(), 1, 3, 0);

    cimg_forXY(image, x, y) {
        // Calculate the grayscale value of the pixel
        unsigned char grayValue = 0.299 * image(x, y, 0) +
                                  0.587 * image(x, y, 1) +
                                  0.114 * image(x, y, 2);

        // Set the grayscale value in the gray image
        grayImage(x, y) = grayValue;
    }

    // Calculate the gradient in the grayscale image
    cimg_forXY(grayImage, x, y) {
        // If the pixel is not at the edge of the image
        if (x > 0 && x < grayImage.width() - 1 && y > 0 &&
            y < grayImage.height() - 1) {
            // Calculate the gradient in the x and y directions
            int gradientX = grayImage(x + 1, y) - grayImage(x - 1, y);
            int gradientY = grayImage(x, y + 1) - grayImage(x, y - 1);

            // Calculate the magnitude of the gradient
            int magnitude = sqrt(gradientX * gradientX + gradientY * gradientY);

            // Set the pixel value in the edge image
            edge(x, y) = magnitude;
        }
    }

    return edge;
}

/**
 * Calculate gradient separately in RGB dimension and combine
 */
int gradientInColor(CImg &image, CImg &gradient) {}
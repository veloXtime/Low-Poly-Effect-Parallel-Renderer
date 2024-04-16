#include "edgedraw.h"

#include <iostream>

#include "CImg.h"

/**
 * Extract edges from the image using Canny edge detection method.
 *
 * @param image Image to extract edge, with RGB color and single depth.
 * @param method Method to use for edge detection, 0 for grayscale, 1 for RGB.
 * @pre Noise should have been removed on the image in previous steps.
 */
CImg extractEdgeCanny(CImg &image, int method) {
    // Create a new image to store the edge
    CImgGradient gradient(image.width(), image.height());

    // Calculate gradient magnitude for each pixel
    if (method == 0) {
        gradientInGray(image, gradient);
    } else {
        gradientInColor(image, gradient);
    }

    // TODO: Non-maximum Suppression
    return image;
}

/**
 * Convert colored image to grayscale and calculate gradient
 */
void gradientInGray(CImg &image, CImgGradient &gradient) {
    // Convert the image to grayscale
    CImg grayImage(image.width(), image.height(), 1, 3, 0);

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
            gradient(x, y) = calculateGradient(grayImage, x, y);
        }
    }
}

/**
 * Calculate gradient separately in RGB dimension and combine
 */
void gradientInColor(CImg &image, CImgGradient &gradient) {
    // TODO: Implement this function
    std::cout << "Error: Function not implemented" << std::endl;
}

/**
 * Calculate gradient for a single pixel
 *
 * @return Gradient magnitude of the pixel
 * @pre The pixel is not at the edge of the image.
 */
Gradient calculateGradient(CImg &image, int x, int y) {
    static const int SOBEL_X[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    static const int SOBEL_Y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    // Calculate the gradient in the x and y directions
    int gradientX = 0;
    int gradientY = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            gradientX += SOBEL_X[i + 1][j + 1] * image(x + i, y + j);
            gradientY += SOBEL_Y[i + 1][j + 1] * image(x + i, y + j);
        }
    }

    // Calculate the magnitude of the gradient
    return Gradient(sqrt(gradientX * gradientX + gradientY * gradientY),
                    atan2(gradientY, gradientX));
}
#include "edgedraw.h"

#include <iostream>

#include "CImg.h"
#include "processing.h"

/**
 * Extract edges from the image using Canny edge detection method.
 *
 * @param image Image to extract edge, with RGB color and single depth.
 * @param method Method to use for edge detection, 0 for grayscale, 1 for RGB.
 * @pre Noise should have been removed on the image in previous steps.
 */
CImg extractEdgeCanny(CImg &image, int method) {
    // Create a new image to store the edge
    CImg gradient(image.width(), image.height());
    CImgFloat direction(image.width(), image.height());

    // Calculate gradient magnitude for each pixel
    if (method == 0) {
        gradientInGray(image, gradient, direction);
    } else {
        gradientInColor(image, gradient, direction);
    }

    // print the gradient
    // for (int i = 0; i < gradient.width(); i++) {
    //     for (int j = 0; j < gradient.height(); j++) {
    //         std::cout << "Gradient at (" << i << ", " << j
    //                   << "): " << gradient(i, j).mag << "\t"
    //                   << gradient(i, j).dir << std::endl;
    //     }
    // }

    // Non-maximum Suppression
    CImg edge(image.width(), image.height());
    nonMaxSuppression(edge, gradient, direction);

    // Track edges
    trackEdge(edge);

    return edge;
}

/**
 * Convert colored image to grayscale and calculate gradient
 */
void gradientInGray(CImg &image, CImg &gradient, CImgFloat &direction) {
    // Convert the image to grayscale
    CImg grayImage(image.width(), image.height(), 1, 1, 0);

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
            gradientResp gr = calculateGradient(grayImage, x, y);
            gradient(x, y) = gr.mag;
            direction(x, y) = gr.dir;
        }
    }
}

/**
 * Calculate gradient separately in RGB dimension and combine
 */
void gradientInColor(CImg &image, CImg &gradient, CImgFloat &direction) {
    // TODO: Implement this function
    std::cout << "Error: Function not implemented" << std::endl;
}

/**
 * Calculate gradient for a single pixel
 *
 * @return Gradient magnitude of the pixel
 * @pre The pixel is not at the edge of the image.
 */
gradientResp calculateGradient(CImg &image, int x, int y) {
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
    return gradientResp(sqrt(gradientX * gradientX + gradientY * gradientY),
                        atan2(gradientY, gradientX) * 180 / M_PI);
}

/**
 * Apply non-maximum suppression to the gradient image
 */
void nonMaxSuppression(CImg &edge, CImg &gradient, CImgFloat &direction) {
    cimg_forXY(edge, x, y) {
        // If the pixel is not at the edge of the image
        if (x > 0 && x < edge.width() - 1 && y > 0 && y < edge.height() - 1) {
            if (x > 0 && x < edge.width() - 1 && y > 0 &&
                y < edge.height() - 1) {
                float angle = direction(x, y);  // Get the continuous angle
                int dir = discretizeDirection(
                    angle);  // Discretize the angle into four main directions

                unsigned char magnitude = gradient(x, y);
                unsigned char mag1 = 0, mag2 = 0;

                // Determine neighboring pixels to compare based on the gradient
                // direction
                switch (dir) {
                    case 0:  // Horizontal edge (East-West)
                        mag1 = gradient(x - 1, y);
                        mag2 = gradient(x + 1, y);
                        break;
                    case 1:  // Diagonal edge (Northeast-Southwest)
                        mag1 = gradient(x - 1, y - 1);
                        mag2 = gradient(x + 1, y + 1);
                        break;
                    case 2:  // Vertical edge (North-South)
                        mag1 = gradient(x, y - 1);
                        mag2 = gradient(x, y + 1);
                        break;
                    case 3:  // Diagonal edge (Northwest-Southeast)
                        mag1 = gradient(x + 1, y - 1);
                        mag2 = gradient(x - 1, y + 1);
                        break;
                }

                // Retain pixel if its magnitude is greater than its neighbors
                // along the gradient direction
                if (magnitude > mag1 && magnitude > mag2) {
                    edge(x, y) = magnitude;  // This pixel is a local maximum
                } else {
                    edge(x, y) = 0;  // Suppress pixel
                }
            }
        }
    }
}

int discretizeDirection(float angle) {
    if ((angle < 22.5 && angle >= -22.5) || angle >= 157.5 || angle < -157.5) {
        return 0;
    } else if ((angle >= 22.5 && angle < 67.5) ||
               (angle < -112.5 && angle >= -157.5)) {
        return 1;
    } else if ((angle >= 67.5 && angle < 112.5) ||
               (angle < -67.5 && angle >= -112.5)) {
        return 2;
    } else if ((angle >= 112.5 && angle < 157.5) ||
               (angle < -22.5 && angle >= -67.5)) {
        return 3;
    } else {
        std::cout << "Error: Angle out of range" << std::endl;
        return -1;
    }
}

void trackEdge(CImg &edge) {
    const unsigned char highThreshold = 100;
    const unsigned char lowThreshold = 10;

    cimg_forXY(edge, x, y) {
        if (edge(x, y) >= highThreshold && edge(x, y) != 255) {
            // Mark as a strong edge
            mark(edge, x, y, lowThreshold);
        } else if (edge(x, y) < lowThreshold) {
            edge(x, y) = 0;  // Suppress noise
        }
    }
}

void mark(CImg &edge, int x, int y, unsigned char lowThreshold) {
    edge(x, y) = 255;  // Mark as a strong edge

    // Check 8-connected neighbors for weak edges
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < edge.width() && ny >= 0 && ny < edge.height() &&
                edge(nx, ny) != 255 && edge(nx, ny) >= lowThreshold) {
                // Recursively mark weak edges
                mark(edge, nx, ny, lowThreshold);
            }
        }
    }
}
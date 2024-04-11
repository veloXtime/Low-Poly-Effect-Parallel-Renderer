#include <iostream>

#include "CImg.h"

cimg_library::CImg<unsigned char> extractEdge(
    cimg_library::CImg<unsigned char> &image) {
    // Create a new image to store the edge
    cimg_library::CImg<unsigned char> edge(image.width(), image.height(), 1, 1,
                                           0);

    // Calculate gradient magnitude for each point
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
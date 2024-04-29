#include <iostream>

#include "CImg.h"
#include "processing.h"

using CImgBool = cimg_library::CImg<boolean>;
const int ANCHOR_THRESH = 8;


int isHorizontal(float angle) {
    if (angle < 45 && angle >= -45 || angle >= 136 || angle < -135) {
        return 0; // horizontal
    } else {
        return 1;
    }
}

/**
 * Test if a pixel is an anchor
 */
void isAnchor(CImg &gradient, CImgFloat &direction, CImgBool &anchor) {
    cimg_forXY(anchor, x, y) {
        // If the pixel is not at the edge of the image
        if (x > 0 && x < anchor.width() - 1 && y > 0 && y < anchor.height() - 1) {
            if (x > 0 && x < anchor.width() - 1 && y > 0 && y < anchor.height() - 1) {
                float angle = direction(x, y);  // Get the continuous angle
                unsigned char magnitude = gradient(x, y);
                unsigned char mag1 = 0, mag2 = 0;

                if (isHorizontal(angle)) {
                    mag1 = gradient(x - 1, y);
                    mag2 = gradient(x + 1, y);
                } else {
                    mag1 = gradient(x, y - 1);
                    mag2 = gradient(x, y + 1);
                }

                // Retain pixel if its magnitude is greater than its neighbors
                // along the gradient direction
                if (magnitude - mag1 >= ANCHOR_THRESH
		    && magnitude - mag2 >= ANCHOR_THRESH) {
                    anchor(x, y) = true;  // This pixel is a local maximum
                } else {
                    anchor(x, y) = false;  // Suppress pixel
                }
            }
        }
    }
}

void routeEdge(CImg &image, CImg &gradient, CImg &edge, CImgFloat &direction,CImgBool &anchor) {
   cimg_forXY(image, x, y) {
        // If pixel (x,y) is an anchor and the direction is horizontal
            if (gradient(x, y) != 0 && edge(x,y) == 0 && isHorizontal(direction(x, y))) {
            int currentX = x;
            int currentY = y;

            // Go left until the edge ends or we reach the image border
            while (currentX > 0 && gradient(currentX, currentY) > 0 && !edge(currentX, currentY)) {
                edge(currentX, currentY) = true;  // Mark this pixel as part of an edge

                // Look at 3 neighbors to the left and pick the one with the max gradient value
                float leftUp = gradient(currentX - 1, currentY - 1);
                float left = gradient(currentX - 1, currentY);
                float leftDown = gradient(currentX - 1, currentY + 1);

                // Move to the pixel with the highest gradient value
                if (leftUp > left && leftUp > leftDown) {
                    currentX -= 1;
                    currentY -= 1; // Move up-left
                } else if (leftDown > left && leftDown > leftUp) {
                    currentX -= 1;
                    currentY += 1; // Move down-left
                } else {
                    currentX -= 1; // Move straight-left
                }
            }
            // Go right until the edge ends or we reach the image border

        }
    } 
}

void drawEdge(CImg &edge) {
    // Calculate mean and standard deviation of the gradient magnitude
    unsigned int sum = 0;
    unsigned int sumSq = 0;
    unsigned int numPixels = edge.width() * edge.height();

    cimg_forXY(edge, x, y) { sum += edge(x, y); }
    float mean = sum / numPixels;

    cimg_forXY(edge, x, y) {
        sumSq += (edge(x, y) - mean) * (edge(x, y) - mean);
    }
    float stdDev = sqrt(sumSq / numPixels);

    // Calculate high and low thresholds according to mean and standard
    // deviation
    unsigned char highThreshold = mean + 2 * stdDev;
    unsigned char lowThreshold = mean + 1 * stdDev;

    cimg_forXY(edge, x, y) {
        if (edge(x, y) >= highThreshold && edge(x, y) != 255) {
            // Mark as a strong edge
            mark(edge, x, y, lowThreshold);
        } else if (edge(x, y) < lowThreshold) {
            edge(x, y) = 0;  // Suppress noise
        }
    }

    // Clear unselected edges
    cimg_forXY(edge, x, y) {
        if (edge(x, y) != 255) {
            edge(x, y) = 0;
        }
    }
}

CImg edgeDraw(CImg &image, int method) {
    // Create a new image to store the edge
    CImg gradient(image.width(), image.height());
    CImgFloat direction(image.width(), image.height());

    // Calculate gradient magnitude for each pixel
    if (method == 0) {
        gradientInGray(image, gradient, direction);
    } else {
        gradientInColor(image, gradient, direction);
    }

    // Non-maximum Suppression
    CImg edge(image.width(), image.height());
    CImgBool anchors(image.width(), image.height());

    // Find anchors
    isAnchor(gradient, direction, anchor);

    // Route edge

    // Track edges
    trackEdge(edge);

    return edge;
}

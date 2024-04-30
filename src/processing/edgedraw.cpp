#include <iostream>

#include "CImg.h"
#include "processing.h"

const int ANCHOR_THRESH = 8;
const int GRADIENT_THRESH = 30;

/**
 * Suppress weak gradients in the image by setting pixels below a certain
 * threshold to zero.
 * @param gradient The gradient image where the suppression is applied.
 */
void suppressWeakGradients(CImg &gradient) {
    cimg_forXY(gradient, x, y) {
        if (gradient(x, y) <= GRADIENT_THRESH) {
            gradient(x, y) = 0;
        }
    }
}

/**
 * Mark edge points in the gradient image.
 * @param gradient The gradient image used to mark edge points.
 * @param edge The binary image where edges will be marked as true.
 */
void markEdgePoints(CImg &gradient, CImgBool &edge) {
    cimg_forXY(gradient, x, y) {
        if (edge(x, y)) {
            gradient(x, y) = 255;
        } else {
            gradient(x, y) = 0;
        }
    }
}

/**
 * Check if the gradient direction at a pixel is approximately horizontal.
 * @param angle The angle of the gradient.
 * @return Returns 1 if horizontal, otherwise 0.
 */
int isHorizontal(float angle) {
    if (angle < 45 && angle >= -45 || angle >= 136 || angle < -135) {
        return 1;  // horizontal
    } else {
        return 0;
    }
}

// Check if a valid coordinate on the image
bool valid(int x, int y, int width, int height) {
    return !(x < 0 || y < 0 || x >= width || y >= height);
}

/**
 * Validate if coordinates (x, y) are within the image bounds.
 * @param x X-coordinate.
 * @param y Y-coordinate.
 * @param width Width of the image.
 * @param height Height of the image.
 * @return True if valid, false otherwise.
 */
bool valid(int x, int y, int width, int height) {
    return !(x < 0 || y < 0 || x >= width || y >= height);
}

/**
 * Determine and mark anchor points in the image based on gradient magnitude and
 * direction.
 * @param gradient The gradient magnitudes.
 * @param direction The gradient directions.
 * @param anchors Binary image to mark anchors.
 */
void determineAnchors(CImg &gradient, CImgFloat &direction, CImgBool &anchor) {
    cimg_forXY(anchor, x, y) {
        // If the pixel is not at the edge of the image
        anchor(x, y) = false;
        if (x > 0 && x < anchor.width() - 1 && y > 0 &&
            y < anchor.height() - 1 && x % 2 == 0 && y % 2 == 0) {
            float angle = direction(x, y);  // Get the continuous angle
            unsigned char magnitude = gradient(x, y);
            unsigned char mag1 = 0, mag2 = 0;

            if (isHorizontal(angle)) {
                mag1 = gradient(x, y - 1);
                mag2 = gradient(x, y + 1);
            } else {
                mag1 = gradient(x - 1, y);
                mag2 = gradient(x + 1, y);
            }

            // Retain pixel if its magnitude is greater than its neighbors
            // along the gradient direction
            if (magnitude - mag1 >= ANCHOR_THRESH &&
                magnitude - mag2 >= ANCHOR_THRESH) {
                anchor(x, y) = true;  // This pixel is a local maximum
            }
        }
    }
}

/**
 * Process edge drawing from a given anchor point.
 * @param x X-coordinate of the anchor.
 * @param y Y-coordinate of the anchor.
 * @param gradient The gradient image.
 * @param direction The gradient directions.
 * @param edge Binary image to mark edges.
 * @param isHorizontal Flag to indicate if the edge should be drawn horizontally
or vertically.
 */
void drawEdgeFromAnchor(int x, int y, CImg<unsigned char> &gradient,
                        const CImg<float> &direction, CImg<bool> &edge,
                        bool isHorizontal) {
    int dx[3], dy[3];  // direction vectors
    if (isHorizontal) {
        dx[0] = -1;
        dy[0] = -1;  // left-up
        dx[1] = -1;
        dy[1] = 0;  // left
        dx[2] = -1;
        dy[2] = 1;  // left-down
    } else {
        dx[0] = -1;
        dy[0] = -1;  // up-left
        dx[1] = 0;
        dy[1] = -1;  // up
        dx[2] = 1;
        dy[2] = -1;  // up-right
    }

    int width = gradient.width(), height = gradient.height();
    int curr_x = x, curr_y = y;

    // Trace in both directions from the anchor point
    for (int dir = -1; dir <= 1; dir += 2) {  // -1 for backward, 1 for forward
        curr_x = x;
        curr_y = y;
        while (isValid(curr_x, curr_y, width, height) &&
               !edge(curr_x, curr_y) && gradient(curr_x, curr_y) > 0) {
            edge(curr_x, curr_y) = true;
            int best_dx = 0, best_dy = 0;
            float max_gradient = 0;

            // Choose the direction with the highest gradient magnitude
            for (int i = 0; i < 3; i++) {
                int nx = curr_x + dir * dx[i], ny = curr_y + dir * dy[i];
                if (isValid(nx, ny, width, height) &&
                    gradient(nx, ny) > max_gradient) {
                    max_gradient = gradient(nx, ny);
                    best_dx = dx[i];
                    best_dy = dy[i];
                }
            }

            curr_x += dir * best_dx;
            curr_y += dir * best_dy;
        }
    }
}

/**
 * Initiate edge drawing from any anchor points.
 * @param gradient The gradient image.
 * @param direction The gradient directions.
 * @param anchors Binary image with only anchor points set to true.
 * @param edge Binary image waiting for edges to be marked as true.
 */
void drawEdgesFromAnchors(const CImg<unsigned char> &gradient,
                          const CImg<float> &direction,
                          const CImg<bool> &anchors, CImg<bool> &edge) {
    cimg_forXY(anchors, x, y) {
        if (anchors(x, y)) {
            drawEdgeFromAnchor(x, y, gradient, direction, edge,
                               isHorizontal(direction(x, y)));
        }
    }
}

/**
 * Main function to perform edge detection on an image.
 * @param image Input image.
 * @param method Method to compute the gradient (0 for grayscale, 1 for color).
 * @return Image containing edges.
 */
CImg edgeDraw(const CImg &image, int method) {
    // Create a new image to store the edge
    CImg gradient(image.width(), image.height());
    CImgFloat direction(image.width(), image.height());
    // Calculate gradient magnitude for each pixel
    if (method == 0) {
        gradientInGray(image, gradient, direction);
    } else {
        gradientInColor(image, gradient, direction);
    }
    suppressWeakGradients(gradient);

    CImgBool edge(image.width(), image.height(), 1, 1, false);
    CImgBool anchor(image.width(), image.height(), 1, 1, false);

    // Find anchors and draw edges from anchors
    determineAnchors(gradient, direction, anchor);
    drawEdgesFromAnchors(gradient, direction, anchors, edge);

    // Only take edge points
    markEdgePoints(gradient, edge);

    getEdgePoints(gradient, edge);

    return gradient;
}

#include "edgedraw.h"

#include <iostream>

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
 * Check if the gradient direction at a pixel is approximately horizontal.
 * @param angle The angle of the gradient.
 * @return Returns 1 if horizontal, otherwise 0.
 */
bool isHorizontal(float angle) {
    if (angle < 45 && angle >= -45 || angle >= 136 || angle < -135) {
        return true;  // horizontal
    } else {
        return false;
    }
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
    return x > 0 && y > 0 && x < width - 1 && y < height - 1;
}

/**
 * Determine and mark anchor points in the image based on gradient magnitude and
 * direction.
 * @param gradient The gradient magnitudes.
 * @param direction The gradient directions.
 * @param anchors Binary image to mark anchors.
 */
void determineAnchors(const CImg &gradient, const CImgFloat &direction,
                      CImgBool &anchor) {
    cimg_forXY(anchor, x, y) {
        // If the pixel is not at the edge of the image
        anchor(x, y) = false;
        if (x > 0 && x < anchor.width() - 1 && y > 0 &&
            y < anchor.height() - 1 ) {
            float angle = direction(x, y);  // Get the continuous angle
            int magnitude = gradient(x, y);
            int mag1 = 0, mag2 = 0;

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
 * Draw horizontal edges from an anchor which direction is horizontal
 * @param x X-coordinate of the anchor.
 * @param y Y-coordinate of the anchor.
 * @param gradient The gradient image.
 * @param direction The gradient directions.
 * @param edge Binary image to mark edges.
 */
void drawHorizontalEdgeFromAnchor(int x, int y, const CImg &gradient,
                                  const CImgFloat &direction, CImg &edge) {
    int width = gradient.width();
    int height = gradient.height();
    if (!valid(x, y, width, height) || edge(x, y)) return;

    int curr_x = x;
    int curr_y = y;
    edge(x, y) = 0;
    while (valid(curr_x, curr_y, width, height) &&
           gradient(curr_x, curr_y) > 0 && !edge(curr_x, curr_y) &&
           isHorizontal(direction(curr_x, curr_y))) {
        edge(curr_x, curr_y) = 255;
        unsigned char leftUp = gradient(curr_x - 1, curr_y - 1);
        unsigned char left = gradient(curr_x - 1, curr_y);
        unsigned char leftDown = gradient(curr_x - 1, curr_y + 1);
        // Move to the pixel with the highest gradient value
        if (leftUp > left && leftUp > leftDown) {
            curr_x -= 1;
            curr_y -= 1;  // Move up-left
        } else if (leftDown > left && leftDown > leftUp) {
            curr_x -= 1;
            curr_y += 1;  // Move down-left
        } else {
            curr_x -= 1;  // Move straight-left
        }
    }
    drawEdgesFromAnchor(curr_x, curr_y, gradient, direction, edge, false);

    curr_x = x;
    curr_y = y;
    edge(x, y) = 0;
    while (valid(curr_x, curr_y, width, height) &&
           gradient(curr_x, curr_y) > 0 && !edge(curr_x, curr_y) &&
           isHorizontal(direction(curr_x, curr_y))) {
        edge(curr_x, curr_y) = 255;
        unsigned char rightUp = gradient(curr_x - 1, curr_y - 1);
        unsigned char right = gradient(curr_x - 1, curr_y);
        unsigned char rightDown = gradient(curr_x - 1, curr_y + 1);
        // Move to the pixel with the highest gradient value
        if (rightUp > right && rightUp > rightDown) {
            curr_x += 1;
            curr_y -= 1;  // Move up-right
        } else if (rightDown > right && rightDown > rightUp) {
            curr_x += 1;
            curr_y += 1;  // Move down-right
        } else {
            curr_x += 1;  // Move straight-right
        }
    }
    drawEdgesFromAnchor(curr_x, curr_y, gradient, direction, edge, false);
}

/**
 * Draw vertical edges from an anchor which direction is horizontal
 * @param x X-coordinate of the anchor.
 * @param y Y-coordinate of the anchor.
 * @param gradient The gradient image.
 * @param direction The gradient directions.
 * @param edge Binary image to mark edges.
 */
void drawVerticalEdgeFromAnchor(int x, int y, const CImg &gradient,
                                const CImgFloat &direction, CImg &edge) {
    int width = gradient.width();
    int height = gradient.height();
    if (!valid(x, y, width, height)) return;

    int curr_x = x;
    int curr_y = y;
    edge(x, y) = 0;  // Assuming white edges on a black background

    // Trace upwards from the anchor point
    while (valid(curr_x, curr_y, width, height) &&
           gradient(curr_x, curr_y) > 0 && !edge(curr_x, curr_y) &&
           !isHorizontal(direction(curr_x, curr_y))) {
        edge(curr_x, curr_y) = 255;  // Mark this pixel as part of an edge
        unsigned char upLeft = gradient(curr_x - 1, curr_y - 1);
        unsigned char up = gradient(curr_x, curr_y - 1);
        unsigned char upRight = gradient(curr_x + 1, curr_y - 1);

        // Move to the pixel with the highest gradient value above the current
        // pixel
        if (upLeft > up && upLeft > upRight) {
            curr_x -= 1;
            curr_y -= 1;  // Move top-left
        } else if (upRight > up && upRight > upLeft) {
            curr_x += 1;
            curr_y -= 1;  // Move top-right
        } else {
            curr_y -= 1;  // Move straight up
        }
    }
    drawEdgesFromAnchor(curr_x, curr_y, gradient, direction, edge, true);

    // Reset to anchor point
    curr_x = x;
    curr_y = y;
    edge(x, y) = 0;

    // Trace downwards from the anchor point
    while (valid(curr_x, curr_y, width, height) &&
           gradient(curr_x, curr_y) > 0 && !edge(curr_x, curr_y) &&
           !isHorizontal(direction(curr_x, curr_y))) {
        edge(curr_x, curr_y) = 255;  // Mark this pixel as part of an edge
        unsigned char downLeft = gradient(curr_x - 1, curr_y + 1);
        unsigned char down = gradient(curr_x, curr_y + 1);
        unsigned char downRight = gradient(curr_x + 1, curr_y + 1);

        // Move to the pixel with the highest gradient value below the current
        // pixel
        if (downLeft > down && downLeft > downRight) {
            curr_x -= 1;
            curr_y += 1;  // Move bottom-left
        } else if (downRight > down && downRight > downLeft) {
            curr_x += 1;
            curr_y += 1;  // Move bottom-right
        } else {
            curr_y += 1;  // Move straight down
        }
    }
    drawEdgesFromAnchor(curr_x, curr_y, gradient, direction, edge, true);
}

/**
 * Recursively draw edge from any an anchor point
 *
 * Say if anchor point direction is horizontal, then we need to proceed both to
 * the left and to the right. Say if we proceed left first, we keep selecting
 * immediate left neighbors which have largest gradient and horizontal
 * direction. When we reach a point which has a vertical direction, then split
 * again at this point and both proceed up and down. The procedure continues
 * until we reach a point which either has 0 gradient or is an edge point.
 *
 * @param x The anchor point X-Coordinate
 * @param y The anchor point Y-Coordinate
 * @param gradient The gradient image.
 * @param direction The gradient directions.
 * @param edge Binary image waiting for edges to be marked as true.
 */
void drawEdgesFromAnchor(int x, int y, const CImg &gradient,
                         const CImgFloat &direction, CImg &edge,
                         const bool isHorizontal) {
    // Check recursion base condition
    if (!valid(x, y, gradient.width(), gradient.height()) ||
        gradient(x, y) <= 0 || edge(x, y)) {
        return;
    }

    if (isHorizontal) {
        drawHorizontalEdgeFromAnchor(x, y, gradient, direction, edge);
    } else {
        drawVerticalEdgeFromAnchor(x, y, gradient, direction, edge);
    }
}

/**
 * Initiate edge drawing from any anchor points.
 * @param gradient The gradient image.
 * @param direction The gradient directions.
 * @param anchors Binary image with only anchor points set to true.
 * @param edge Binary image waiting for edges to be marked as true.
 */
void drawEdgesFromAnchors(const CImg &gradient, const CImgFloat &direction,
                          const CImgBool &anchors, CImg &edge) {
    cimg_forXY(anchors, x, y) {
        if (anchors(x, y)) {
            drawEdgesFromAnchor(x, y, gradient, direction, edge,
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
    suppressWeakGradients(gradient);

    CImg edge(image.width(), image.height(), 1, 1, 0);
    CImgBool anchor(image.width(), image.height(), 1, 1, false);

    // Find anchors and draw edges from anchors
    determineAnchors(gradient, direction, anchor);
    drawEdgesFromAnchors(gradient, direction, anchor, edge);

    // return edge;
    return edge;
}

CImg edgeDrawGPU(CImg &image, int method) {
    // Create a new image to store the edge
    CImg gradient(image.width(), image.height());
    CImgFloat direction(image.width(), image.height());
    // Calculate gradient magnitude for each pixel
    if (method == 0) {
        gradientInGrayGPU(image, gradient, direction);
    } else {
        gradientInColor(image, gradient, direction);
    }
    suppressWeakGradientsGPU(gradient);

    CImg edge(image.width(), image.height(), 1, 1, 0);
    CImgBool anchor(image.width(), image.height(), 1, 1, false);

    // Find anchors and draw edges from anchors
    determineAnchorsGPU(gradient, direction, anchor);
    drawEdgesFromAnchors(gradient, direction, anchor, edge);

    // return edge;
    return edge;
}

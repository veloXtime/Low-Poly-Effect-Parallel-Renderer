#include <iostream>

#include "CImg.h"
#include "processing.h"

const int ANCHOR_THRESH = 8;
const int GRADIENT_THRESH = 40;

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
            y < anchor.height() - 1 && x % 4 == 0 && y % 4 == 0) {
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
 * Draw horizontal edges from an anchor which direction is horizontal
 * @param x X-coordinate of the anchor.
 * @param y Y-coordinate of the anchor.
 * @param gradient The gradient image.
 * @param direction The gradient directions.
 * @param edge Binary image to mark edges.
 */
void drawHorizontalEdgeFromAnchor(int x, int y, const CImg &gradient,
                                  const CImgFloat &direction, CImgBool &edge) {
    std::cout << "horizontal " << x << " " << y << std::endl;

    int width = gradient.width();
    int height = gradient.height();
    std::cout << "point 0" << std::endl;
    if (!valid(x, y, width, height) || edge(x, y)) return;

    int curr_x = x;
    int curr_y = y;
    edge(x, y) = false;
    std::cout << "point 1" << std::endl;
    while (valid(curr_x, curr_y, width, height) &&
           gradient(curr_x, curr_y) > 0 && !edge(curr_x, curr_y) &&
           isHorizontal(direction(curr_x, curr_y))) {
        std::cout << "point 2" << std::endl;
        edge(curr_x, curr_y) = true;
        float leftUp = gradient(curr_x - 1, curr_y - 1);
        float left = gradient(curr_x - 1, curr_y);
        float leftDown = gradient(curr_x - 1, curr_y + 1);
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
    edge(x, y) = false;
    while (valid(curr_x, curr_y, width, height) &&
           gradient(curr_x, curr_y) > 0 && !edge(curr_x, curr_y) &&
           isHorizontal(direction(curr_x, curr_y))) {
        edge(curr_x, curr_y) = true;
        float rightUp = gradient(curr_x - 1, curr_y - 1);
        float right = gradient(curr_x - 1, curr_y);
        float rightDown = gradient(curr_x - 1, curr_y + 1);
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
                                const CImgFloat &direction, CImgBool &edge) {
    std::cout << "vertical " << x << " " << y << std::endl;
    int width = gradient.width();
    int height = gradient.height();
    std::cout << "point 8" << std::endl;
    if (!valid(x, y, width, height)) return;

    int curr_x = x;
    int curr_y = y;
    edge(x, y) = false;  // Assuming white edges on a black background
    std::cout << "point 9" << std::endl;

    // Trace upwards from the anchor point
    while (valid(curr_x, curr_y, width, height) &&
           gradient(curr_x, curr_y) > 0 && !edge(curr_x, curr_y) &&
           !isHorizontal(direction(curr_x, curr_y))) {
        std::cout << "point 10" << std::endl;
        edge(curr_x, curr_y) = true;  // Mark this pixel as part of an edge
        float upLeft = gradient(curr_x - 1, curr_y - 1);
        float up = gradient(curr_x, curr_y - 1);
        float upRight = gradient(curr_x + 1, curr_y - 1);

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
    edge(x, y) = false;

    // Trace downwards from the anchor point
    while (valid(curr_x, curr_y, width, height) &&
           gradient(curr_x, curr_y) > 0 && !edge(curr_x, curr_y) &&
           !isHorizontal(direction(curr_x, curr_y))) {
        edge(curr_x, curr_y) = true;  // Mark this pixel as part of an edge
        float downLeft = gradient(curr_x - 1, curr_y + 1);
        float down = gradient(curr_x, curr_y + 1);
        float downRight = gradient(curr_x + 1, curr_y + 1);

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
 * Initiate edge drawing from any anchor points.
 * @param gradient The gradient image.
 * @param direction The gradient directions.
 * @param anchors Binary image with only anchor points set to true.
 * @param edge Binary image waiting for edges to be marked as true.
 */
void drawEdgesFromAnchor(int x, int y, const CImg &gradient,
                         const CImgFloat &direction, CImgBool &edge,
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
                          const CImgBool &anchors, CImgBool &edge) {
    cimg_forXY(anchors, x, y) {
        if (anchors(x, y)) {
            drawEdgesFromAnchor(x, y, gradient, direction, edge, isHorizontal(direction(x, y)));
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

    CImgBool edge(image.width(), image.height(), 1, 1, false);
    CImgBool anchor(image.width(), image.height(), 1, 1, false);

    // Find anchors and draw edges from anchors
    determineAnchors(gradient, direction, anchor);
    drawEdgesFromAnchors(gradient, direction, anchor, edge);

    // Only take edge points
    markEdgePoints(gradient, edge);

    return gradient;
}

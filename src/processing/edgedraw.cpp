#include <iostream>

#include "CImg.h"
#include "processing.h"

using CImg = cimg_library::CImg<unsigned char>;
using CImgBool = cimg_library::CImg<boolean>;
const int ANCHOR_THRESH = 8;
const int GRADIENT_THRESH = 100;

/** Apply a threshold to the gradient map to suppress weak gradients. */
CImg suppressWeakGradients(const CImg &gradient) {
    cimg_forXY(gradient, x, y) {
        if (gradient(x, y) > GRADIENT_THRESH) {
            gradient(x, y) = 255;  // Consider as strong edge (retain)
        } else {
            gradient(x, y) = 0;  // Consider as weak edge (suppress)
        }
    }
}

/** Check if the direction of a gradient is horizontal */
int isHorizontal(float angle) {
    if (angle < 45 && angle >= -45 || angle >= 136 || angle < -135) {
        return 1;  // horizontal
    } else {
        return 0;
    }
}

/**
 * Test if a pixel is an anchor, set anchor(x,y) to true.
 */
void isAnchor(CImg &gradient, CImgFloat &direction, CImgBool &anchor) {
    cimg_forXY(anchor, x, y) {
        // If the pixel is not at the edge of the image
        if (x > 0 && x < anchor.width() - 1 && y > 0 &&
            y < anchor.height() - 1) {
            if (x > 0 && x < anchor.width() - 1 && y > 0 &&
                y < anchor.height() - 1) {
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
                if (magnitude - mag1 >= ANCHOR_THRESH &&
                    magnitude - mag2 >= ANCHOR_THRESH) {
                    anchor(x, y) = true;  // This pixel is a local maximum
                } else {
                    anchor(x, y) = false;  // Suppress pixel
                }
            }
        }
    }
}

void drawHorizontalEdgeFromAnchor(int x, int y, CImg &gradient,
                                  CImgFloat &direction, CImgBool &edge) {
    int curr_x = x;
    int curr_y = y;
    edge(x, y) = false;
    while (gradient(curr_x, curr_y) > 0 && !edge(curr_x, curr_y) &&
           isHorizontal(direction(x, y))) {
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

    int curr_x = x;
    int curr_y = y;
    edge(x, y) = false;
    while (gradient(curr_x, curr_y) > 0 && !edge(curr_x, curr_y) &&
           isHorizontal(direction(x, y))) {
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
}

void drawVerticalEdgeFromAnchor(int x, int y, CImg<unsigned char> &gradient,
                                CImgFloat &direction,
                                CImg<unsigned char> &edge) {
    int curr_x = x;
    int curr_y = y;
    edge(x, y) = 255;  // Assuming white edges on a black background

    // Trace upwards from the anchor point
    while (gradient(curr_x, curr_y) > 0 && !edge(curr_x, curr_y) &&
           !isHorizontal(direction(curr_x, curr_y))) {
        edge(curr_x, curr_y) = 255;  // Mark this pixel as part of an edge
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

    // Reset to anchor point
    curr_x = x;
    curr_y = y;

    // Trace downwards from the anchor point
    while (gradient(curr_x, curr_y) > 0 && !edge(curr_x, curr_y) &&
           !isHorizontal(direction(curr_x, curr_y))) {
        edge(curr_x, curr_y) = 255;  // Mark this pixel as part of an edge
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
}

void drawEdgeFromAnchor(int x, int y, CImg &gradient, CImgFloat &direction,
                        CImg &edge) {
    // Starting at the anchor, if left
    if (isHorizontal(direction(x, y))) {
        drawHorizontalEdgeFromAnchor(x, y, gradient, direction, edge);
    } else {
        drawVerticalEdgeFromAnchor(x, y, gradient, direction, edge);
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
    CImgBool edge(image.width(), image.height());
    CImgBool anchors(image.width(), image.height());

    // Find anchors
    isAnchor(gradient, direction, anchor);

    // Route edge
    suppressWeakGradients(gradient);

    // Track edges
    cimg_forXY(anchor, x, y) {
        if (anchor(x, y)) {
            drawEdgeFromAnchor(x, y, gradient, direction, edge);
        }
    }

    return gradient;
}

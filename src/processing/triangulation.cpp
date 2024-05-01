#include <iostream>

#include "CImg.h"
#include "processing.h"

/**
 * Pick a subset of points in edges for triangulation
 * @param edge The edge obtained from edge draw algorithm
 */
void pickVertices(CImg &edge) {
    cimg_forXY(edge, x, y) {
        if (x % 4 != 0 || y % 4 != 0) {
            edge(x, y) = 0;
        }
    }
}



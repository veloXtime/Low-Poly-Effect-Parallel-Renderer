#ifndef DELAUNAY_H 
#define DELAUNAY_H

#include "CImg.h"

using CImg = cimg_library::CImg<unsigned char>;

// Functions for Delaunay triangulation
void pickVertices(CImg &edge);

#endif
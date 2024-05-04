#ifndef DELAUNAY_H 
#define DELAUNAY_H

#include "CImg.h"

struct Point {
    int x;
    int y;
};

struct Color {
    int R;
    int G;
    int B;
};

struct Triangle {
    int s1;
    int s2;
    int s3;
};

using CImg = cimg_library::CImg<unsigned char>;
using CImgInt = cimg_library::CImg<int>;

// Functions for Delaunay triangulation
void pickVertices(CImg &edge);

CImgInt jumpFloodAlgorithm(CImg& vertices);
void delaunayTriangulation(CImgInt &voronoi, CImg &image);

CImg colorVoronoiDiagram(CImgInt &voronoi);

#endif
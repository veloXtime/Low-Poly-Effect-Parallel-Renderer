#include <iostream>
#include <random>
#include <set>
#include <unordered_map>
#include <algorithm>

#include "delaunay.h"

// @todo: change to use siteId = x * width + y to store site center information

/**
 * Pick a subset of points in edges for triangulation
 * @param edge The edge obtained from edge draw algorithm
 */
void pickVertices(CImg &edge) {
    int ctr = 1;
    cimg_forXY(edge, x, y) {
        if (x % 4 != 0 || y % 4 != 0) {
            edge(x, y) = 0;
        }
    }

    edge(0, 0) = 255;
    edge(0, edge.height() - 1) = 255;
    edge(edge.width() - 1, 0) = 255;
    edge(edge.width() - 1, edge.height() - 1) = 255;
}

int squaredDistance(int x, int y, int xx, int yy) {
    if (x == -1 || y == -1 || xx == -1 || y == -1) {
        return -1;
    } else {
        int dx = x - xx;
        int dy = y - yy;
        return dx * dx + dy * dy;
    }
}

CImgInt jumpFloodAlgorithm(CImg &vertices) {
    int width = vertices.width();
    int height = vertices.height();

    // Our voronoi diagram which each pixel contains
    // information about closest site/vertex
    CImgInt voronoi(width, height, 1, 1, -1);
    cimg_forXY(vertices, x, y) {
        if (vertices(x, y) != 0) {
            voronoi(x, y) = x * width + y;
        }
    }

    int maxStep = std::max(vertices.width(), vertices.height()) / 2;
    while (maxStep > 0) {
        cimg_forXY(voronoi, x, y) {
            int minSiteId = -1;
            int minDist = -1;  // squared distance (x1 - x2)^2 + (y1 - y2)^2
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = x + dx * maxStep;
                    int ny = y + dy * maxStep;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        int siteId = voronoi(nx, ny);
                        if (siteId != -1) {
                            int dist = squaredDistance(x, y, siteId % width,
                                                       siteId / width);
                            if (minDist == -1 || minDist > dist) {
                                minSiteId = siteId;
                                minDist = dist;
                            }
                        }
                    }
                }
            }
            voronoi(x, y) = minSiteId;
        }
        maxStep /= 2;  // Reduce the step size
    }

    return voronoi;
}

CImg colorVoronoiDiagram(CImgInt &voronoi) {
    int width = voronoi.width();
    int height = voronoi.height();
    CImg coloredVoronoi(width, height, 1, 2, 0);
    std::uniform_int_distribution<int> distribution(0, 255);
    std::default_random_engine generator;

    std::unordered_map<int, Color> siteColor;

    cimg_forXY(coloredVoronoi, x, y) {
        int siteId = voronoi(x, y);
        if (siteColor.find(siteId) == siteColor.end()) {
            Color newColor =
                Color{distribution(generator), distribution(generator),
                      distribution(generator)};
            siteColor[siteId] = newColor;
        }

        Color c = siteColor[siteId];
        coloredVoronoi(x, y, 0) = c.R;
        coloredVoronoi(x, y, 1) = c.G;
        coloredVoronoi(x, y, 2) = c.B;
    }

    return coloredVoronoi;
}

bool pointInTriangle(int px, int py, int ax, int ay, int bx, int by, int cx, int cy) {
    int v0x = cx - ax;
    int v0y = cy - ay;
    int v1x = bx - ax;
    int v1y = by - ay;
    int v2x = px - ax;
    int v2y = py - ay;

    float dot00 = v0x * v0x + v0y * v0y;
    float dot01 = v0x * v1x + v0y * v1y;
    float dot02 = v0x * v2x + v0y * v2y;
    float dot11 = v1x * v1x + v1y * v1y;
    float dot12 = v1x * v2x + v1y * v2y;

    float invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
    float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    return (u >= 0) && (v >= 0) && (u + v < 1);
}

void delaunayTriangulation(CImgInt &voronoi, CImg &image) {
    int width = voronoi.width();
    int height = voronoi.height();
    std::vector<Triangle> triangles;

    cimg_forXY(voronoi, x, y) {
        if (x < width - 1 && y < height - 1) {
            int topLeft = voronoi(x, y);
            int topRight = voronoi(x + 1, y);
            int botLeft = voronoi(x, y + 1);
            int botRight = voronoi(x + 1, y + 1);

            std::set<int> uniqueSites;
            uniqueSites.insert(topLeft);
            uniqueSites.insert(topRight);
            uniqueSites.insert(botLeft);
            uniqueSites.insert(botRight);
            std::vector<int> sitesVector(uniqueSites.begin(),
                                         uniqueSites.end());

            if (sitesVector.size() == 4) {
                triangles.push_back(Triangle{topLeft, topRight, botLeft});
                triangles.push_back(Triangle{topRight, botLeft, botRight});
            } else if (sitesVector.size() == 3) {
                triangles.push_back(Triangle{sitesVector[0], sitesVector[1],
                                    sitesVector[2]});
            }
        }
    }

    for (int i = 0; i < triangles.size(); i++) {
        int s1 = triangles[i].s1;
        int s2 = triangles[i].s2;
        int s3 = triangles[i].s3;

        int ax = s1 % width, ay = s1 / width;
        int bx = s2 % width, by = s2 / width;
        int cx = s3 % width, cy = s3 / width;

        float R = 0;
        float G = 0;
        float B = 0;
        float ctr = 0;

        // Scan over image dimensions
        for (int x = std::min(std::min(ax, bx), cx); x < std::max(std::max(ax, bx), cx); x++) {
            for (int y = std::min(std::min(ay, by), cy); y < std::max(std::max(ay, by), cy); y++) {
                if (pointInTriangle(x, y, ax, ay, bx, by, cx, cy)) {
                    R += image(x, y, 0);
                    G += image(x, y, 1);
                    B += image(x, y, 2);
                    ctr++;
                }
            }
        }
        if (ctr > 0) {
        R = R / ctr;
        G = G / ctr;
        B = B / ctr; }
        for (int x = std::min(std::min(ax, bx), cx); x < std::max(std::max(ax, bx), cx); x++) {
            for (int y = std::min(std::min(ay, by), cy); y < std::max(std::max(ay, by), cy); y++) {
                if (pointInTriangle(x, y, ax, ay, bx, by, cx, cy)) {
                    image(x, y, 0) = int(R);
                    image(x, y, 1) = int(G);
                    image(x, y, 2) = int(B);
                }
            }
        }
    }
}
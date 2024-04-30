#ifndef PROCESSING_H
#define PROCESSING_H
#include "CImg.h"

using CImg = cimg_library::CImg<unsigned char>;
using CImgBool = cimg_library::CImg<bool>;
using CImgFloat = cimg_library::CImg<float>;

// Functions for Gaussian blur
unsigned char *gaussianBlurCPU(const unsigned char *inputImage, int width,
                               int height, int channels);
unsigned char *gaussianBlur(const unsigned char *inputImage, int width,
                            int height, int channels);

// Functions for edge detection and edge draw
struct gradientResp {
    float mag;  // magnitude of gradient
    float dir;  // direction of the gradient

    gradientResp(float _mag, float _dir) : mag(_mag), dir(_dir) {}
};
void gradientInGray(CImg &image, CImg &gradient, CImgFloat &direction);
void gradientInColor(CImg &image, CImg &gradient, CImgFloat &direction);
gradientResp calculateGradient(CImg &image, int x, int y);
void nonMaxSuppression(CImg &edge, CImg &gradient, CImgFloat &direction);
int discretizeDirection(float angle);
void trackEdge(CImg &edge);
void mark(CImg &edge, int x, int y, unsigned char lowThreshold);
void drawEdgesFromAnchor(int x, int y, const CImg &gradient,
                         const CImgFloat &direction, CImgBool &edge,
                         const bool isHorizontal);
void drawHorizontalEdgeFromAnchor(int x, int y, const CImg &gradient,
                                  const CImgFloat &direction, CImgBool &edge);
void drawVerticalEdgeFromAnchor(int x, int y, const CImg &gradient,
                                const CImgFloat &direction, CImgBool &edge);
cimg_library::CImg<unsigned char> extractEdge(
    cimg_library::CImg<unsigned char> &image);
cimg_library::CImg<unsigned char> extractEdgeCanny(
    cimg_library::CImg<unsigned char> &image, int method = 0);
cimg_library::CImg<unsigned char> edgeDraw(
    cimg_library::CImg<unsigned char> &image, int method = 0);

#endif
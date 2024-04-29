#include "CImg.h"

using CImg = cimg_library::CImg<unsigned char>;
using CImgFloat = cimg_library::CImg<float>;

struct gradientResp {
    float mag;
    float dir;

    gradientResp(float _mag, float _dir) : mag(_mag), dir(_dir) {}
};

void gradientInGray(CImg &image, CImg &gradient, CImgFloat &direction);
void gradientInColor(CImg &image, CImg &gradient, CImgFloat &direction);
gradientResp calculateGradient(CImg &image, int x, int y);
void nonMaxSuppression(CImg &edge, CImg &gradient, CImgFloat &direction);
int discretizeDirection(float angle);
void trackEdge(CImg &edge);
void mark(CImg &edge, int x, int y, unsigned char lowThreshold);
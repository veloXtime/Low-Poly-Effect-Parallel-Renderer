#include "CImg.h"

using CImg = cimg_library::CImg<unsigned char>;
using CImgInt = cimg_library::CImg<int>;

struct gradientResp {
    unsigned char mag;
    int dir;

    gradientResp(unsigned char _mag, int _dir) : mag(_mag), dir(_dir) {}
};

void gradientInGray(CImg &image, CImg &edge, CImgInt &direction);
void gradientInColor(CImg &image, CImg &edge, CImgInt &direction);
gradientResp calculateGradient(CImg &image, int x, int y);
void nonMaxSuppression(CImg &edge, CImgInt &direction);
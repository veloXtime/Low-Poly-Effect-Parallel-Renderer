#include "CImg.h"

using CImg = cimg_library::CImg<unsigned char>;
using CImgShort = cimg_library::CImg<short>;

struct gradientResp {
    unsigned char mag;
    short dir;

    gradientResp(unsigned char _mag, short _dir) : mag(_mag), dir(_dir) {}
};

void gradientInGray(CImg &image, CImg &gradient, CImgShort &direction);
void gradientInColor(CImg &image, CImg &gradient, CImgShort &direction);
gradientResp calculateGradient(CImg &image, int x, int y);
void nonMaxSuppression(CImg &edge, CImg &gradient, CImgShort &direction);
int discretizeDirection(short angle);
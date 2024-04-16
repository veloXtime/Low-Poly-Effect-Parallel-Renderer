#include "CImg.h"

struct Gradient {
    int val;
    int dir;

    Gradient(int _val = 0, int _dir = 0) : val(_val), dir(_dir) {}
};

using CImg = cimg_library::CImg<unsigned char>;
using CImgGradient = cimg_library::CImg<Gradient>;

void gradientInGray(CImg &image, CImgGradient &gradient);
void gradientInColor(CImg &image, CImgGradient &gradient);
Gradient calculateGradient(CImg &image, int x, int y);
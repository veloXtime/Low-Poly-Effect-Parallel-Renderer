#include "CImg.h"
#include <iostream>
using namespace cimg_library;
using namespace std;

int main() {
    // Ask the user to enter the path to the image
    cout << "Enter the path to the image: ";
    string imagePath;
    getline(cin, imagePath); // Read the entire line, including spaces

    // Load the image
    CImg<unsigned char> image(imagePath.c_str());

    // Display the image
    CImgDisplay display(image, "Loaded Image");

    // Wait for the display window to close
    while (!display.is_closed()) {
        display.wait();
    }

    return 0;
}

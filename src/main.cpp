#include <chrono>
#include <iostream>

#include "CImg.h"
#include "processing.h"
using namespace cimg_library;
using namespace std;

int main() {
    // Ask the user to enter the path to the image
    cout << "Enter the path to the image: ";
    string imagePath;
    getline(cin, imagePath);  // Read the entire line, including spaces

    // Load the image
    CImg<unsigned char> image(imagePath.c_str());

    // Convert the image to a format compatible with CUDA function
    int width = image.width();
    int height = image.height();
    unsigned char* inputImage = image.data();
    int channels = image.spectrum();

    // Apply Gaussian blur using CUDA
    auto start = chrono::high_resolution_clock::now();
    unsigned char* outputImage =
        gaussianBlur(inputImage, width, height, channels);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Time taken for Gaussian blur: " << duration.count()
         << " microseconds" << endl;

    CImg<unsigned char> blurredImage(outputImage, width, height, 1, 3, true);
    // CImg<unsigned char> blurredImage = image.blur(1.5);

    // Display the original and blurred images
    CImgDisplay display(image, "Original Image");
    CImgDisplay displayBlurred(blurredImage, "Blurred Image");
    // Wait for the display windows to close
    while (!display.is_closed() && !displayBlurred.is_closed()) {
        display.wait();
        displayBlurred.wait();
    }

    // Free memory
    delete[] outputImage;

    return 0;
}
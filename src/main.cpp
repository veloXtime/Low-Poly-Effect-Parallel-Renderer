#include <chrono>
#include <iostream>

#include "CImg.h"
#include "processing.h"
using namespace std;

unsigned char* applyGaussianBlur(cimg_library::CImg<unsigned char>& image) {
    int width = image.width();
    int height = image.height();
    unsigned char* inputImage = image.data();
    int channels = image.spectrum();

    // Apply Gaussian blur using CUDA
    auto start = chrono::high_resolution_clock::now();

    unsigned char* outputImage =
        gaussianBlur(inputImage, width, height, channels);

    auto end = chrono::high_resolution_clock::now();

    // Get duration
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Time taken for Gaussian blur: " << duration.count()
         << " microseconds" << endl;

    return outputImage;
}

unsigned char* applyGaussianBlurCPU(cimg_library::CImg<unsigned char>& image) {
    int width = image.width();
    int height = image.height();
    unsigned char* inputImage = image.data();
    int channels = image.spectrum();

    // Apply Gaussian blur using CUDA
    auto start = chrono::high_resolution_clock::now();

    unsigned char* outputImage =
        gaussianBlurCPU(inputImage, width, height, channels);

    auto end = chrono::high_resolution_clock::now();

    // Get duration
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Time taken for Gaussian blur: " << duration.count()
         << " microseconds" << endl;

    return outputImage;
}

int main(int argc, char* argv[]) {
    string imagePath;

    // Get image path from commandline or cin
    if (argc > 1) {
        imagePath = argv[1];
    } else {
        // Ask the user to enter the path to the image
        cout << "Enter the path to the image: ";
        getline(cin, imagePath);  // Read the entire line, including spaces
    }

    // Load the image
    cimg_library::CImg<unsigned char> image(imagePath.c_str());

    // Step 1: perform the Gaussian blur
    unsigned char* gbImage = applyGaussianBlur(image);
    cimg_library::CImg<unsigned char> blurredImage(gbImage, width, height, 1, 3,
                                                   true);

    // Apply edge extraction using CPU
    start = chrono::high_resolution_clock::now();
    cimg_library::CImg<unsigned char> edge = extractEdgeCanny(blurredImage);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Time taken for Edge Extraction: " << duration.count()
         << " microseconds" << endl;

    // Display the original and blurred images
    cimg_library::CImgDisplay display(image, "Original Image");
    cimg_library::CImgDisplay displayBlurred(blurredImage, "Blurred Image");
    cimg_library::CImgDisplay displayEdge(edge, "Edge Image");
    // Wait for the display windows to close
    while (!display.is_closed() && !displayBlurred.is_closed() &&
           !displayEdge.is_closed()) {
        display.wait();
        displayBlurred.wait();
        displayEdge.wait();
    }

    // Free memory
    delete[] gbImage;

    return 0;
}
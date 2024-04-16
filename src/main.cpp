#include <chrono>
#include <iostream>

#include "CImg.h"
#include "processing.h"
using namespace std;

int main(int argc, char* argv[]) {
    string imagePath;

    // Try to get from commandl line arguments
    if (argc > 1) {
        imagePath = argv[1];
    } else {
        // Ask the user to enter the path to the image
        cout << "Enter the path to the image: ";
        getline(cin, imagePath);  // Read the entire line, including spaces
    }

    // Load the image
    cimg_library::CImg<unsigned char> image(imagePath.c_str());

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

    cimg_library::CImg<unsigned char> blurredImage(outputImage, width, height,
                                                   1, 3, true);
    // CImg<unsigned char> blurredImage = image.blur(1.5);

    cimg_library::CImg<unsigned char> edge = extractEdgeCanny(blurredImage);

    // Display the original and blurred images
    cimg_library::CImgDisplay display(image, "Original Image");
    cimg_library::CImgDisplay displayBlurred(blurredImage, "Blurred Image");
    cimg_library::CImgDisplay displayEdge(edge, "Edge Image");
    // Wait for the display windows to close
    while (!display.is_closed() && !displayBlurred.is_closed()) {
        display.wait();
        displayBlurred.wait();
        displayEdge.wait();
    }

    // Free memory
    delete[] outputImage;

    return 0;
}
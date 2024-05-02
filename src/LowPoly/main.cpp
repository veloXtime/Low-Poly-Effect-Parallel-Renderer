#include <chrono>
#include <iostream>

#include "CImg.h"
#include "delaunay.h"
#include "edgedraw.h"
#include "gaussianblur.h"

using namespace std;

unsigned char* applyGaussianBlur(cimg_library::CImg<unsigned char>& image,
                                 int width, int height, int channels) {
    // Apply Gaussian blur using CUDA
    auto start = std::chrono::high_resolution_clock::now();

    unsigned char* outputImage = gaussianBlur(image, width, height, channels);

    auto end = std::chrono::high_resolution_clock::now();

    // Get duration
    auto duration =
        std::chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Time taken for Gaussian blur: " << duration.count()
         << " microseconds" << endl;

    return outputImage;
}

unsigned char* applyGaussianBlurCPU(cimg_library::CImg<unsigned char>& image,
                                    int width, int height, int channels) {
    // Apply Gaussian blur using CUDA
    auto start = std::chrono::high_resolution_clock::now();

    unsigned char* outputImage =
        gaussianBlurCPU(image, width, height, channels);

    auto end = std::chrono::high_resolution_clock::now();

    // Get duration
    auto duration =
        std::chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Time taken for Gaussian blur CPU: " << duration.count()
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
    int width = image.width();
    int height = image.height();
    unsigned char* inputImage = image.data();
    int channels = image.spectrum();

    // Step 1: perform the Gaussian blur
    unsigned char* gbImage = applyGaussianBlur(image, width, height, channels);
    // gbImage = applyGaussianBlurCPU(image, width, height, channels);
    cimg_library::CImg<unsigned char> blurredImage(gbImage, width, height, 1, 3,
                                                   true);

    // Apply edge extraction using CPU
    auto start = chrono::high_resolution_clock::now();
    cimg_library::CImg<unsigned char> edgeCPU = edgeDraw(blurredImage);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Time taken for Edge Extraction (CPU): " << duration.count()
         << " microseconds" << endl;

    // Apply edge extraction using GPU
    start = chrono::high_resolution_clock::now();
    cimg_library::CImg<unsigned char> edgeGPU = edgeDrawGPU(blurredImage);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Time taken for Edge Extraction (GPU): " << duration.count()
         << " microseconds" << endl;

    // Compare each pixel of the CPU and GPU edge images and print the number
    // and value of mismatches
    int mismatches = 0;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            if (edgeCPU(i, j) != edgeGPU(i, j)) {
                mismatches++;
                cout << "Mismatch at (" << i << ", " << j
                     << "): " << (int)edgeCPU(i, j) << " vs "
                     << (int)edgeGPU(i, j) << endl;
            }
        }
    }

    // Delaunay triangulation
    // pickVertices(edge);

    // Display the original and blurred images
    cimg_library::CImgDisplay display(image, "Original Image");
    cimg_library::CImgDisplay displayBlurred(blurredImage, "Blurred Image");
    cimg_library::CImgDisplay displayEdgeCPU(edgeCPU, "Edge Image CPU");
    cimg_library::CImgDisplay displayEdgeGPU(edgeGPU, "Edge Image GPU");
    // Wait for the display windows to close
    while (!display.is_closed() && !displayBlurred.is_closed() &&
           !displayEdgeCPU.is_closed() && !displayEdgeGPU.is_closed()) {
        display.wait();
        displayBlurred.wait();
        displayEdgeCPU.wait();
        displayEdgeGPU.wait();
    }

    // Free memory
    delete[] gbImage;

    return 0;
}
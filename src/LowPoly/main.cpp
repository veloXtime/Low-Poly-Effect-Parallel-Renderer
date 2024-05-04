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

CImg applyEdgeDetectionCPU(CImg& blurredImage) {
    auto start = chrono::high_resolution_clock::now();
    cimg_library::CImg<unsigned char> edgeCPU = edgeDraw(blurredImage);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Time taken for Edge Extraction (CPU): " << duration.count()
         << " microseconds" << endl;
    return edgeCPU;
}

CImg applyEdgeDetectionGPU(CImg& blurredImage) {
    auto start = chrono::high_resolution_clock::now();
    cimg_library::CImg<unsigned char> edgeGPU = edgeDrawGPU(blurredImage);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Time taken for Edge Extraction (GPU): " << duration.count()
         << " microseconds" << endl;
    return edgeGPU;
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
    CImg image(imagePath.c_str());
    int width = image.width();
    int height = image.height();
    unsigned char* inputImage = image.data();
    int channels = image.spectrum();

    cout << image.width() << " " << image.height() << endl;

    // Step 1: perform the Gaussian blur
    unsigned char* gbImage = applyGaussianBlur(image, width, height, channels);
    // gbImage = applyGaussianBlurCPU(image, width, height, channels);
    CImg blurredImage(gbImage, width, height, 1, 3, true);

    // Apply edge extraction using CPU
    CImg edgeCPU = applyEdgeDetectionCPU(blurredImage);

    // Apply edge extraction using GPU
    // CImg edgeGPU = applyEdgeDetectionGPU(blurredImage);

    // Delaunay triangulation
    pickVertices(edgeCPU);
    CImgInt voronoi = jumpFloodAlgorithm(edgeCPU);
    delaunayTriangulation(voronoi, image);

    // Display the original and blurred images
    cimg_library::CImgDisplay display(image, "Original Image");
    cimg_library::CImgDisplay displayBlurred(blurredImage, "Blurred Image");
    cimg_library::CImgDisplay displayEdgeCPU(edgeCPU, "Edge Image CPU");
    // cimg_library::CImgDisplay displayEdgeGPU(edgeGPU, "Edge Image GPU");
    cimg_library::CImgDisplay displayVoronoi(voronoi, "Edge Image");
    // Wait for the display windows to close
    while (!display.is_closed() && 
           !displayEdgeCPU.is_closed() && !displayVoronoi.is_closed() ) {
        display.wait();
        displayEdgeCPU.wait();
        displayVoronoi.wait();
        // displayEdgeGPU.wait();
    }

    // Free memory
    delete[] gbImage;

    return 0;
}
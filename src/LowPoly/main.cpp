#include <chrono>
#include <iostream>

#include "CImg.h"
#include "delaunay.h"
#include "edgedraw.h"
#include "gaussianblur.h"

using namespace std;

unsigned char* applyGaussianBlur(CImg& image, int width, int height,
                                 int channels) {
    // Apply Gaussian blur using CUDA
    auto start = std::chrono::high_resolution_clock::now();

    unsigned char* outputImage =
        gaussianBlur(image.data(), width, height, channels);

    auto end = std::chrono::high_resolution_clock::now();

    // Get duration
    auto duration =
        std::chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Time taken for Gaussian blur GPU: " << duration.count()
         << " microseconds" << endl;

    return outputImage;
}

unsigned char* applyGaussianBlurCPU(CImg& image, int width, int height,
                                    int channels) {
    // Apply Gaussian blur using CUDA
    auto start = std::chrono::high_resolution_clock::now();

    unsigned char* outputImage =
        gaussianBlurCPU(image.data(), width, height, channels);

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

CImg applyEdgeDetectionGPUCombined(CImg& blurredImage) {
    auto start = chrono::high_resolution_clock::now();
    cimg_library::CImg<unsigned char> edgeGPU =
        edgeDrawGPUCombined(blurredImage);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Time taken for Edge Extraction (GPU Combined): "
         << duration.count() << " microseconds" << endl;
    return edgeGPU;
}

void applyTriangulation(CImg& edge, CImg& image) {
    cout << "-------------------------------------------" << endl;
    auto veryStart = chrono::high_resolution_clock::now();
    pickVertices(edge);
    auto end = chrono::high_resolution_clock::now();
    auto duration =
        chrono::duration_cast<chrono::microseconds>(end - veryStart);
    cout << "Time taken for pickVertices (CPU): " << duration.count()
         << " microseconds" << endl;

    auto start = chrono::high_resolution_clock::now();
    CImgInt voronoi = jumpFloodAlgorithm(edge);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Time taken for jumpFloodAlgorithm (CPU): " << duration.count()
         << " microseconds" << endl;

    start = chrono::high_resolution_clock::now();
    delaunayTriangulation(voronoi, image);
    auto veryEnd = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(veryEnd - start);
    cout << "Time taken for delaunayTriangulation (CPU): " << duration.count()
         << " microseconds" << endl;

    duration = chrono::duration_cast<chrono::microseconds>(veryEnd - veryStart);
    cout << "Time taken for Triangulation in total (CPU): " << duration.count()
         << " microseconds" << endl;
    cout << "-------------------------------------------" << endl;
}

void applyTriangulationGPU(CImg& edge, CImg& image) {
    cout << "-------------------------------------------" << endl;
    auto veryStart = chrono::high_resolution_clock::now();
    pickVerticesGPU(edge);
    auto end = chrono::high_resolution_clock::now();
    auto duration =
        chrono::duration_cast<chrono::microseconds>(end - veryStart);
    cout << "Time taken for pickVertices (GPU): " << duration.count()
         << " microseconds" << endl;

    auto start = chrono::high_resolution_clock::now();
    CImgInt voronoi = jumpFloodAlgorithmGPU(edge);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Time taken for jumpFloodAlgorithm (GPU): " << duration.count()
         << " microseconds" << endl;

    start = chrono::high_resolution_clock::now();
    delaunayTriangulationGPU(voronoi, image);
    auto veryEnd = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(veryEnd - start);
    cout << "Time taken for delaunayTriangulation (GPU): " << duration.count()
         << " microseconds" << endl;

    duration = chrono::duration_cast<chrono::microseconds>(veryEnd - veryStart);
    cout << "Time taken for Triangulation in total (GPU): " << duration.count()
         << " microseconds" << endl;
    cout << "-------------------------------------------" << endl;
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

    gpuWarmUp();

    // Load the image
    CImg image(imagePath.c_str());
    int width = image.width();
    int height = image.height();
    unsigned char* inputImage = image.data();
    int channels = image.spectrum();

    cout << image.width() << " " << image.height() << endl;

    // Step 1: perform the Gaussian blur
    unsigned char* gbImage = applyGaussianBlur(image, width, height, channels);
    gbImage = applyGaussianBlurCPU(image, width, height, channels);

    CImg blurredImage(gbImage, width, height, 1, 3, true);

    // Apply edge extraction using CPU
    CImg edgeCPU = applyEdgeDetectionCPU(blurredImage);

    // Apply edge extraction using GPU
    CImg edgeGPU = applyEdgeDetectionGPU(blurredImage);

    // Apply edge extraction using GPU Combined
    CImg edgeGPUCombined = applyEdgeDetectionGPUCombined(blurredImage);

    // // Delaunay triangulation
    CImg lowPolyImageCPU = image;
    CImg lowPolyImageGPU = image;
    applyTriangulation(edgeCPU, lowPolyImageCPU);
    applyTriangulationGPU(edgeGPU, lowPolyImageGPU);

    // Display the original and blurred images
    cimg_library::CImgDisplay display(image, "Original Image");
    cimg_library::CImgDisplay displayBlurred(blurredImage, "Blurred Image");
    cimg_library::CImgDisplay displayEdgeCPU(edgeCPU, "Edge Image CPU");
    cimg_library::CImgDisplay displayEdgeGPU(edgeGPU, "Edge Image GPU");
    cimg_library::CImgDisplay displayLowPolyCPU(lowPolyImageCPU,
                                                "Low Poly Image CPU");
    cimg_library::CImgDisplay displayLowPolyGPU(lowPolyImageGPU,
                                                "Low Poly Image GPU");

    // cimg_library::CImgDisplay displayVoronoi(voronoi, "Edge Image");
    // Wait for the display windows to close
    while (!display.is_closed() && !displayEdgeCPU.is_closed() &&
           !displayBlurred.is_closed() && !displayEdgeGPU.is_closed() &&
           !displayLowPolyCPU.is_closed()) {
        display.wait();
        displayEdgeCPU.wait();
        displayEdgeGPU.wait();
        displayLowPolyCPU.wait();
        displayLowPolyGPU.wait();
    }

    // Free memory
    delete[] gbImage;

    return 0;
}
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "delaunay.h"

__global__ void pickVerticesKernel(unsigned char *edge, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        if (edge[idx] == 254) {
            edge[idx] = 255;
        } else {
            edge[idx] = 0;
        }
    }
}

void pickVerticesGPU(CImg &edge) {
    int width = edge.width();
    int height = edge.height();

    unsigned char *d_edge;
    size_t size = width * height * sizeof(unsigned char);

    cudaMalloc(&d_edge, size);
    cudaMemcpy(d_edge, edge.data(), size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y);

    pickVerticesKernel<<<dimGrid, dimBlock>>>(d_edge, width, height);

    cudaMemcpy(edge.data(), d_edge, size, cudaMemcpyDeviceToHost);
    cudaFree(d_edge);

    // Set the corners to 255
    edge(0, 0) = 255;
    edge(0, height - 1) = 255;
    edge(width - 1, 0) = 255;
    edge(width - 1, height - 1) = 255;

    // Optionally, add edge boundary points
    std::uniform_int_distribution<int> distribution(0, 255);
    std::default_random_engine generator;
    for (int x = 0; x < width; x += distribution(generator)) {
        for (int y = 0; y < height; y += distribution(generator)) {
            edge(x, y) = 255;
        }
    }
    for (int x = 0; x < width; x += distribution(generator)) {
        edge(x, 0) = 255;
        edge(x, height - 1) = 255;
    }
    for (int y = 0; y < height; y += distribution(generator)) {
        edge(0, y) = 255;
        edge(width - 1, y) = 255;
    }
}

__global__ void setupJumpFloodKernel(unsigned char *d_vertices, int *d_voronoi,
                                     int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        if (d_vertices[idx] != 0) {
            d_voronoi[idx] = idx;
        }
    }
}

__device__ int squaredDistanceCUDA(int x, int y, int xx, int yy) {
    if (x == -1 || y == -1 || xx == -1 || y == -1) {
        return -1;
    } else {
        int dx = x - xx;
        int dy = y - yy;
        return dx * dx + dy * dy;
    }
}

__global__ void jumpFloodAlgorithmKernel(int *d_voronoi, int width, int height,
                                         int step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        int minSiteId = -1;
        int minDist = -1;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = x + dx * step;
                int ny = y + dy * step;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int siteId = d_voronoi[ny * width + nx];
                    if (siteId != -1) {
                        int dist = squaredDistanceCUDA(x, y, siteId % width,
                                                       siteId / width);
                        if (minDist == -1 || minDist > dist) {
                            minSiteId = siteId;
                            minDist = dist;
                        }
                    }
                }
            }
        }
        d_voronoi[idx] = minSiteId;
    }
}

CImgInt jumpFloodAlgorithmGPU(CImg &vertices) {
    int width = vertices.width();
    int height = vertices.height();

    unsigned char *d_vertices;
    int *d_voronoi;
    cudaMalloc(&d_vertices, width * height * sizeof(unsigned char));
    cudaMalloc(&d_voronoi, width * height * sizeof(int));
    cudaMemcpy(d_vertices, vertices.data(),
               width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_voronoi, -1, width * height * sizeof(int));

    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y);

    setupJumpFloodKernel<<<dimGrid, dimBlock>>>(d_vertices, d_voronoi, width,
                                                height);
    cudaFree(d_vertices);

    int maxStep = std::max(width, height) / 2;
    while (maxStep > 0) {
        jumpFloodAlgorithmKernel<<<dimGrid, dimBlock>>>(d_voronoi, width,
                                                        height, maxStep);
        maxStep /= 2;
    }

    CImgInt voronoi(width, height, 1, 1, -1);
    cudaMemcpy(voronoi.data(), d_voronoi, width * height * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaFree(d_voronoi);

    return voronoi;
}

__device__ Triangle TriangleCUDA(int s1, int s2, int s3) {
    Triangle t;
    t.s1 = s1;
    t.s2 = s2;
    t.s3 = s3;
    return t;
}

__global__ void delaunayTriangulationKernel(int *d_voronoi,
                                            unsigned char *d_image,
                                            Triangle *d_triangles,
                                            int *triangles_count, int width,
                                            int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width - 1 && y < height - 1) {
        int topLeft = d_voronoi[y * width + x];
        int topRight = d_voronoi[y * width + x + 1];
        int botLeft = d_voronoi[(y + 1) * width + x];
        int botRight = d_voronoi[(y + 1) * width + x + 1];

        // Count uniques
        int uniques = 0;
        int values[4] = {topLeft, topRight, botLeft, botRight};
        int uniqueValues[4];

        for (int i = 0; i < 4; i++) {
            bool unique = true;
            for (int j = 0; j < i; j++) {
                if (values[i] == values[j]) {
                    unique = false;
                    break;
                }
            }
            if (unique) {
                uniqueValues[uniques] = values[i];
                uniques++;
            }
        }

        if (uniques == 4) {
            int idx = atomicAdd(triangles_count, 2);
            d_triangles[idx] = TriangleCUDA(topLeft, topRight, botLeft);
            d_triangles[idx + 1] = TriangleCUDA(topRight, botLeft, botRight);
        } else if (uniques == 3) {
            int idx = atomicAdd(triangles_count, 1);
            d_triangles[idx] =
                TriangleCUDA(uniqueValues[0], uniqueValues[1], uniqueValues[2]);
        }
    }
}

__device__ bool pointInTriangleCUDA(int px, int py, int ax, int ay, int bx,
                                    int by, int cx, int cy) {
    int v0x = cx - ax;
    int v0y = cy - ay;
    int v1x = bx - ax;
    int v1y = by - ay;
    int v2x = px - ax;
    int v2y = py - ay;

    float dot00 = v0x * v0x + v0y * v0y;
    float dot01 = v0x * v1x + v0y * v1y;
    float dot02 = v0x * v2x + v0y * v2y;
    float dot11 = v1x * v1x + v1y * v1y;
    float dot12 = v1x * v2x + v1y * v2y;

    float invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
    float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    return (u >= 0) && (v >= 0) && (u + v < 1);
}

__device__ Point centerPixelOfTriangleCUDA(int ax, int ay, int bx, int by,
                                           int cx, int cy) {
    int x = (ax + bx + cx) / 3;
    int y = (ay + by + cy) / 3;

    if (pointInTriangleCUDA(x, y, ax, ay, bx, by, cx, cy)) {
        return Point{x, y};
    } else {
        return Point{ax, ay};
    }
}

__global__ void transformTrianglesKernel(unsigned char *d_image,
                                         Triangle *d_triangles,
                                         int triangles_count, int width,
                                         int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < triangles_count) {
        int s1 = d_triangles[idx].s1;
        int s2 = d_triangles[idx].s2;
        int s3 = d_triangles[idx].s3;

        int ax = s1 % width, ay = s1 / width;
        int bx = s2 % width, by = s2 / width;
        int cx = s3 % width, cy = s3 / width;

        Point centerPixel = centerPixelOfTriangleCUDA(ax, ay, bx, by, cx, cy);
        int pixels = width * height;
        int R = d_image[centerPixel.y * width + centerPixel.x];
        int G = d_image[centerPixel.y * width + centerPixel.x + pixels];
        int B = d_image[centerPixel.y * width + centerPixel.x + 2 * pixels];

        // Scan over image dimensions
        for (int x = min(min(ax, bx), cx); x < max(max(ax, bx), cx); x++) {
            for (int y = min(min(ay, by), cy); y < max(max(ay, by), cy); y++) {
                if (pointInTriangleCUDA(x, y, ax, ay, bx, by, cx, cy)) {
                    d_image[y * width + x] = R;
                    d_image[y * width + x + pixels] = G;
                    d_image[y * width + x + 2 * pixels] = B;
                }
            }
        }
    }
}

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort) exit(code);
    }
}

void delaunayTriangulationGPU(CImgInt &voronoi, CImg &image) {
    int width = voronoi.width();
    int height = voronoi.height();

    int *d_voronoi;
    unsigned char *d_image;
    Triangle *d_triangles;
    int *triangles_count;

    size_t voronoiSize = width * height * sizeof(int);
    size_t imageSize = width * height * 3 * sizeof(unsigned char);
    size_t trianglesSize = width * height * sizeof(Triangle);

    cudaMalloc(&d_voronoi, voronoiSize);
    cudaMalloc(&d_image, imageSize);
    cudaMalloc(&d_triangles, trianglesSize);
    cudaMalloc(&triangles_count, sizeof(int));

    cudaMemcpy(d_voronoi, voronoi.data(), voronoiSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_image, image.data(), imageSize, cudaMemcpyHostToDevice);
    cudaMemset(d_triangles, 0, trianglesSize);
    cudaMemset(triangles_count, 0, sizeof(int));

    // Step 1: Find triangles
    dim3 dimBlock1(16, 16);
    dim3 dimGrid1((width + dimBlock1.x - 1) / dimBlock1.x,
                  (height + dimBlock1.y - 1) / dimBlock1.y);
    delaunayTriangulationKernel<<<dimGrid1, dimBlock1>>>(
        d_voronoi, d_image, d_triangles, triangles_count, width, height);

    // Fetch triangles count
    int host_triangles_count;
    cudaMemcpy(&host_triangles_count, triangles_count, sizeof(int),
               cudaMemcpyDeviceToHost);
    std::cout << "\tTriangles count: " << host_triangles_count << std::endl;

    // Step 2: Transform triangles to image
    dim3 dimBlock2(16);
    dim3 dimGrid2((host_triangles_count + dimBlock2.x - 1) / dimBlock2.x);
    transformTrianglesKernel<<<dimGrid2, dimBlock2>>>(
        d_image, d_triangles, host_triangles_count, width, height);

    cudaMemcpy(image.data(), d_image, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_voronoi);
    cudaFree(d_image);
    cudaFree(d_triangles);
}
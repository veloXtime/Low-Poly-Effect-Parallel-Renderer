# Low-Poly-Effect-Parallel-Renderer

A low-poly effect renderer parallelized using C++ and CUDA.

<div style="display: flex; justify-content: flex-start;">
    <img src="images/original.png" alt="Original Image" width="200" height="200"/>
    <img src="images/low-poly.png" alt="Low-poly Image" width="200" height="200"/>
</div>

## Installation
1. **Set up the environment.** Make sure you have NVIDIA CUDA C/C++ Compiler (NVCC) and CUDA shared library installed. Then make sure your environment variables for CUDA are set. For example, if your NVCC is located at `/usr/local/cuda-12.5/bin` and your CUDA shared library is located at `/usr/local/cuda-12.5/lib64`, then run
    ```sh
    export PATH=/usr/local/cuda-11.7/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH ```
2. **Clone the repository.**
    ```sh
    git clone git@github.com:veloXtime/Low-Poly-Effect-Parallel-Renderer.git
    ```
3. **Build the project.** Navigate to the `src/LowPoly/` directory, then run `make`.

## Usage
1. **Run the main executable.** Supply your image path as command line argument. 
    ```sh
    ./main <input_image_path> 
    ```
This will display the low-poly image and print the processing times for different stages on both CPU and GPU to the terminal.

**Note**: Ensure X11 support is enabled on your system to view the output images.


## Reports
See our design and result analysis, including before-and-after images and performance results, at [Low-Poly-Effect-Parallel-Renderer](https://veloxtime.github.io/Low-Poly-Effect-Parallel-Renderer/).
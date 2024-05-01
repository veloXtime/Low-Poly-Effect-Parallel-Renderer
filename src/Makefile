# Compiler
CXX=g++ -m64
CXXFLAGS=-O3 
LDFLAGS=-L/usr/local/cuda-11.7/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc
INCLUDE := -I. -I/CImg.h
# Libraries
LIBS := -lpthread -lX11

# Main executable
main: main.o gaussianblur.o edgedetect.o edgedraw.o triangulation.o
	$(NVCC) $(CXXFLAGS) -o main main.o gaussianblur.o edgedetect.o edgedraw.o triangulation.o $(INCLUDE) $(LIBS)

# Object files
main.o: main.cpp processing.h
	$(CXX) $(CXXFLAGS) -c main.cpp $(INCLUDE)

gaussianblur.o: processing/gaussianblur.cu processing.h
	$(NVCC) $(CXXFLAGS) -c processing/gaussianblur.cu $(INCLUDE)

edgedetect.o: processing/edgedetect.cpp processing.h 
	$(CXX) $(CXXFLAGS) -c processing/edgedetect.cpp $(INCLUDE)

edgedraw.o: processing/edgedraw.cpp processing.h 
	$(CXX) $(CXXFLAGS) -c processing/edgedraw.cpp $(INCLUDE)

triangulation.o: processing/triangulation.cpp processing.h 
	$(CXX) $(CXXFLAGS) -c processing/triangulation.cpp $(INCLUDE)

# Clean
clean:
	rm -f main main.o gaussianblur.o edgedetect.o edgedraw.o

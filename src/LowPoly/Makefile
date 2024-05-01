# Compiler settings
CXX=g++ -m64
CXXFLAGS=-O3 
LDFLAGS=-L/usr/local/cuda-11.7/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc
INCLUDE := -I. -ICImg.h -IDelaunay -IEdgeDraw -IGaussianBlur 
# Libraries
LIBS := -lpthread -lX11

# Main executable
main: main.o gaussianblur.o edgedetect.o edgedraw.o triangulation.o
	$(NVCC) $(NVCCFLAGS) -o main main.o gaussianblur.o edgedetect.o edgedraw.o triangulation.o $(LDFLAGS) $(INCLUDE) $(LIBS)

# Object files
main.o: main.cpp 
	$(CXX) $(CXXFLAGS) -c main.cpp $(INCLUDE)

gaussianblur.o: GaussianBlur/gaussianblur.cu GaussianBlur/gaussianblur.h
	$(NVCC) $(NVCCFLAGS) -c GaussianBlur/gaussianblur.cu $(INCLUDE)

edgedetect.o: EdgeDraw/edgedetect.cpp EdgeDraw/edgedraw.h
	$(CXX) $(CXXFLAGS) -c EdgeDraw/edgedetect.cpp $(INCLUDE)

edgedraw.o: EdgeDraw/edgedraw.cpp EdgeDraw/edgedraw.h
	$(CXX) $(CXXFLAGS) -c EdgeDraw/edgedraw.cpp $(INCLUDE)

triangulation.o: Delaunay/triangulation.cpp Delaunay/delaunay.h
	$(CXX) $(CXXFLAGS) -c Delaunay/triangulation.cpp $(INCLUDE)

# Clean
clean:
	rm -f main main.o gaussianblur.o edgedetect.o edgedraw.o triangulation.o

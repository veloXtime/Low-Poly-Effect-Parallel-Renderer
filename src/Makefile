# EXECUTABLE := cudaScan
# CU_FILES   := scan.cu
# CU_DEPS    :=
# CC_FILES   := main.cpp


CXX = g++
CXXFLAGS = -Wall -std=c++11 -I.
LIBS = -lX11 -lpthread
TARGET = main
SRC = main.cpp edge/edge_draw.cpp
OBJ = $(SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LIBS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean

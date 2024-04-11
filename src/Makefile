# EXECUTABLE := cudaScan
# CU_FILES   := scan.cu
# CU_DEPS    :=
# CC_FILES   := main.cpp


CXX = g++
CXXFLAGS = -Wall -std=c++11
TARGET = image_loader
SRC = main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean

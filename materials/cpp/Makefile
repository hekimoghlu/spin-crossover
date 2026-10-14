# Builds the OpenMP kernel library used by scmaterials.kernels (ctypes).
CXX ?= g++
CXXFLAGS ?= -O3 -march=native -fopenmp -fPIC -std=c++17 -Wall -Wextra
BUILD := build
LIB := $(BUILD)/libscmk.so

all: $(LIB)

$(LIB): scmk.cpp
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -shared -o $@ $<

clean:
	rm -rf $(BUILD)

.PHONY: all clean

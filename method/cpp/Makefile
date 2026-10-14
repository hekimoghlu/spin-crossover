CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -fopenmp -fPIC -Wall -Wextra
all: libdmsim.so libkernels.so
libdmsim.so: dmsim.cpp
	$(CXX) $(CXXFLAGS) -shared -o $@ $<
libkernels.so: kernels.cpp
	$(CXX) $(CXXFLAGS) -shared -o $@ $<
clean:
	rm -f libdmsim.so libkernels.so
.PHONY: all clean

CXX = g++
CXXFLAGS = -O3 -std=c++17 -fopenmp -Wall

all: spmv

spmv: spmv.cpp
	$(CXX) $(CXXFLAGS) -o spmv.hw spmv.cpp

clean:
	rm -f spmv.hw

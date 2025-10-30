CXX = g++
CXXFLAGS = -O3 -std=c++17

all: to_csr

to_csr: to_csr.cpp
	$(CXX) $(CXXFLAGS) -o to_csr.hw to_csr.cpp

clean:
	rm -f to_csr.hw

CXX = g++
CXXFLAGS = -O3 -std=c++17

all: mtx_to_csr

mtx_to_csr: mtx_to_csr.cpp
	$(CXX) $(CXXFLAGS) -o mtx_to_csr.hw mtx_to_csr.cpp

clean:
	rm -f mtx_to_csr.hw

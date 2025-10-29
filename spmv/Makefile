CXX = g++
CXXFLAGS = -O3 -std=c++17 -fopenmp -Wall -no-pie -Wl,-E -rdynamic
LDFLAGS =  -no-pie -Wl,-E

M5OPS_HEADER_PATH=/home/ubuntu/gem5/include
M5_BUILD_PATH=/home/ubuntu/gem5/util/m5/build/arm64/

OUTPUT_SUFFIX=.hw
PICKLEDEVICE_SUFFIX=.pdev
GEM5_SUFFIX=.m5

CXX_FLAGS += -lpickledevice -Wfatal-errors
LDFLAGS   += -lpickledevice
ifeq ($(ENABLE_PICKLEDEVICE), 1)
  CXX_FLAGS += -DENABLE_PICKLEDEVICE=1
  LDFLAGS   += -DENABLE_PICKLEDEVICE=1
  TMP_OUTPUT_SUFFIX = $(OUTPUT_SUFFIX)
  OUTPUT_SUFFIX := $(TMP_OUTPUT_SUFFIX)$(PICKLEDEVICE_SUFFIX)
endif

ifeq ($(ENABLE_GEM5), 1)
  CXX_FLAGS += -I$(M5OPS_HEADER_PATH) -I$(M5OPS_HEADER_PATH)/../util/m5/src
  CXX_FLAGS += -DENABLE_GEM5=1
  LDFLAGS   += -lm5 -L$(M5_BUILD_PATH)/out/
  LDFLAGS   += -DENABLE_GEM5=1
  TMP_OUTPUT_SUFFIX = $(OUTPUT_SUFFIX)
  OUTPUT_SUFFIX := $(TMP_OUTPUT_SUFFIX)$(GEM5_SUFFIX)
endif

all: spmv

spmv: spmv.cpp
	$(CXX) $(CXXFLAGS) -o spmv.hw spmv.cpp

spmv_pdev: spmv.cpp
	$(CXX) $(CXXFLAGS) -o spmv$(OUTPUT_SUFFIX) spmv.cpp  $(LDFLAGS)

clean:
	rm -f spmv.hw*

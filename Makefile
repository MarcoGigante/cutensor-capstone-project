# Makefile for CUDA Capstone Project: Matrix Multiplication with cuTensor (Windows edition)

CXX = nvcc
SRC = cuda_cutensor_matrixmul.cu
EXE = cuda_cutensor_matrixmul.exe

# Detect your own CUDA version and install location if necessary!
CUDA_PATH = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"
WIN_KITS_INC = "C:/Program Files (x86)/Windows Kits/10/Include/10.0.19041.0"
WIN_KITS_LIB = "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.19041.0"

# Compiler flags - add required Windows SDK headers
CXXFLAGS = -std=c++17 -arch=sm_89 \
-I"C:/Program Files (x86)/Windows Kits/10/Include/10.0.19041.0/um" \
-I"C:/Program Files (x86)/Windows Kits/10/Include/10.0.19041.0/shared" \
-I"C:/Program Files (x86)/Windows Kits/10/Include/10.0.19041.0/ucrt" \
-I"C:/Program Files (x86)/Windows Kits/10/Include/10.0.19041.0/winrt" \
-I"C:/Program Files (x86)/Windows Kits/10/Include/10.0.19041.0/cppwinrt" \
-I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include"

# Link against CUDA runtime and cuTensor library - add their actual library path!
LDFLAGS = -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/lib/x64" -lcudart -lcuda -lcutensor -lcublas \
-L"C:/Program Files (x86)/Windows Kits/10/Lib/10.0.19041.0/um/x64" \
-L"C:/Program Files (x86)/Windows Kits/10/Lib/10.0.19041.0/ucrt/x64"

# Default target: clean and build
all: clean build

# Build the executable from the source file
build: $(SRC)
	$(CXX) $(SRC) $(CXXFLAGS) -o bin/$(EXE) $(LDFLAGS)

# Run the executable with optional command-line arguments
run:
	bin/$(EXE) $(ARGS)

# Clean build artifacts and output evidence files (del for Windows!)
clean:
	@if exist bin\$(EXE) del bin\$(EXE)
	@if exist results\matrix_output.txt del results\matrix_output.txt



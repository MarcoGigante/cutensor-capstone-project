#!/usr/bin/env bash
# ------------------------------------------------------------------------
# run.sh - Build and execute CUDA Capstone Matrix Multiplication (cuTensor)
# ------------------------------------------------------------------------

# Clean previous builds and output evidence files
make clean

# Build executable
make build

# Run the program
make run

# Display the resulting evidence file for peer review
echo "---- BEGIN OUTPUT (first 10 elements of result matrix) ----"
cat results/matrix_output.txt
echo "---- END OUTPUT ----"

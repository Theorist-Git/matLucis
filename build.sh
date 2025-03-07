#!/usr/bin/sh

set -xe

# Compiler and flags
CXX=g++
CXXFLAGS="-std=c++11 -fopenmp -Wall -Wextra -Werror -pedantic -O3"

# Output executable
OUT="bin/test"

# Create bin directory if it doesn't exist
mkdir -p bin

echo "Compiling test.cpp -> $OUT"
# Compile testMatrix.cpp to an object file
$CXX $CXXFLAGS -c test/test.cpp -o bin/test.o

# Link the object file(s) into an executable
$CXX $CXXFLAGS bin/test.o -o $OUT

# Check if build succeeded
if [ $? -eq 0 ]; then
    echo "Build successful. Running tests..."
    ./$OUT
else
    echo "Build failed."
    exit 1
fi
        
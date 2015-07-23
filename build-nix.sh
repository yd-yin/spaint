#! /bin/bash -e

# Check that a valid build type has been specified.
if [ $# -ne 2 ] || ([ "$1" != "Unix Makefiles" ] && [ "$1" != "Xcode" ]) || ([ $2 != "Debug" ] && [ $2 != "Release" ])
then
  echo "Usage: build-nix.sh {Unix Makefiles|Xcode} {Debug|Release}"
  exit
fi

# Detect whether this is being run on Linux or Mac OS X.
PLATFORM=linux
if [ "$(uname)" == "Darwin" ]
then
  PLATFORM=mac
fi

# Build/extract the libraries.
cd libraries

./build-boost_1_56_0-nix.sh
#./build-opencv-2.4.9-nix.sh
./build-SDL2-2.0.3-nix.sh
./extract-Eigen-3.2.2.sh

if [ $PLATFORM == "linux" ]
then
  ./build-glew-1.12.0-nix.sh
fi

cd ..

# Build spaint itself.
echo "[spaint] Building spaint"

if [ ! -d build ]
then
  mkdir build
  cd build

  # Note: We need to configure twice to handle conditional building.
  echo "[spaint] ...Configuring using CMake..."
  cmake -G"$1" -DCMAKE_BUILD_TYPE=$2 ..
  cmake ..
  cd ..
fi

cd build

echo "[spaint] ...Running build..."
make -j2

echo "[spaint] ...Installing..."
make install

echo "[spaint] ...Finished building spaint."

#! /bin/bash -e

PLATFORM=`../detect-platform.sh`

TOOLSET=gcc
if [ $PLATFORM == "mac" ]
then
  TOOLSET=darwin
  OSXVERSION=`../detect-osxversion.sh`
fi

# Build Boost 1.58.0.
LOG=../build-boost_1_58_0.log

echo "[spaint] Building Boost 1.58.0"

if [ -d boost_1_58_0 ]
then
  echo "[spaint] ...Skipping build (already built)"
  exit
fi

if [ -d boost-setup ]
then
  echo "[spaint] ...Skipping archive extraction (already extracted)"
else
  mkdir -p setup/boost_1_58_0

  if [ ! -f setup/boost_1_58_0/boost_1_58_0.tar.gz ]
  then
    echo "[spaint] ...Downloading archive..."
    curl -sL http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz > setup/boost_1_58_0/boost_1_58_0.tar.gz
  fi

  echo "[spaint] ...Extracting archive..."
  /bin/rm -fR tmp
  mkdir tmp
  cd tmp
  tar xzf ../setup/boost_1_58_0/boost_1_58_0.tar.gz
  cd ..
  mv tmp/boost_1_58_0 boost-setup
  rmdir tmp
fi

cd boost-setup

if [ -e b2 ]
then
  echo "[spaint] ...Skipping bootstrapping (b2 already exists)"
else
  # Note: This fix is needed when building Boost on a system with Python 3.
  echo "[spaint] ...Fixing bootstrap script..."
  perl -i -pe 's/print sys.prefix/print(sys.prefix)/g' bootstrap.sh

  echo "[spaint] ...Bootstrapping..."
  ./bootstrap.sh > $LOG
fi

echo "[spaint] ...Running build..."
if [ $PLATFORM == "mac" ]
then
  if [ "$OSXVERSION" -ge 13 ]
  then
    STDLIBFLAGS='cxxflags="-stdlib=libstdc++" linkflags="-stdlib=libstdc++"'
  fi
else
  STDLIBFLAGS='cxxflags="-std=c++11"'
fi

./b2 -j2 --libdir=../boost_1_58_0/lib --includedir=../boost_1_58_0/include --abbreviate-paths --with-chrono --with-date_time --with-filesystem --with-program_options --with-regex --with-serialization --with-test --with-thread --with-timer --build-type=complete --layout=tagged toolset=$TOOLSET architecture=x86 address-model=64 $STDLIBFLAGS install >> $LOG

echo "[spaint] ...Fixing headers..."
perl -i -pe 's/SPT<void>/SPT<const void>/g' ../boost_1_58_0/include/boost/serialization/shared_ptr_helper.hpp

echo "[spaint] ...Finished building Boost 1.58.0."

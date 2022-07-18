#! /bin/bash -e

# Check that devenv and msbuild are on the system path.
../require-devenv.sh
../require-msbuild.sh

LOG=../../../build-glew-1.12.0.log

# Check that valid parameters have been specified.
SCRIPT_NAME=`basename "$0"`

if [ $# -ne 1 ] || ([ "$1" != "12" ] && [ "$1" != "15" ])
then
  echo "Usage: $SCRIPT_NAME {12|15}"
  exit 1
fi

# Determine the CMake generator to use.
CMAKE_GENERATOR=`../determine-cmakegenerator.sh $1`

# Build glew.
echo "[spaint] Building glew 1.12.0 for $CMAKE_GENERATOR"

if [ -d glew-1.12.0 ]
then
  echo "[spaint] ...Skipping archive extraction (already extracted)"
else
  echo "[spaint] ...Extracting archive..."
  /bin/rm -fR tmp
  mkdir tmp
  cd tmp
  tar xzf ../setup/glew-1.12.0/glew-1.12.0.tgz
  cd ..
  mv tmp/glew-1.12.0 .
  rmdir tmp
fi

cd glew-1.12.0/build/vc12

if [ $1 == "15" ]
then
  if [ ! -f UpgradeLog.htm ]
  then
    echo "[spaint] ...Upgrading solution..."
    cmd //c "devenv /upgrade glew.sln > $LOG 2>&1"
    result=`cmd //c "(vsdevcmd && set) | grep 'WindowsSDKVersion' | perl -pe 's/.*=(.*)./\1/g'"`
    ls *.vcxproj | while read f; do perl -ibak -pe 's/<ProjectGuid>\{(.*?)\}<\/ProjectGuid>/<ProjectGuid>\{\1\}<\/ProjectGuid>\r    <WindowsTargetPlatformVersion>'$result'<\/WindowsTargetPlatformVersion>/g' "$f"; perl -ibak -pe 's/v141/v140/g' "$f"; done
  else
    echo "[spaint] ...Skipping solution upgrade (already upgraded)"
  fi
fi

echo "[spaint] ...Running build..."
cmd //c "msbuild /p:Configuration=Release /p:Platform=x64 glew.sln >> $LOG 2>&1"

cd ..

echo "[spaint] ...Finished building glew-1.12.0."

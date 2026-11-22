#!/bin/bash
# Private CPython 3.12 macOS arm64 overlay. Pass the benchmark Python executable.
set -euo pipefail
python=${1:?usage: setup_gtsam.sh /path/to/benchmark/python [private_directory]}
root=${2:-"$HOME/agent-share/pcgym-four-library-timing/gtsam"}
source_dir=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$root"
cd "$root"
printf 'gtsam==4.3.0\nnumpy==2.5.1\n' > requirements.in
printf 'cmeel-boost==1.87.0.1\ncmake==3.31.10\n' > build.in
for pair in 'requirements.in requirements.txt' 'build.in build.lock'; do
  read -r spec lock <<< "$pair"
  if [[ ! -f "$lock" ]]; then
    uv pip compile --python "$python" --generate-hashes "$spec" -o "$lock" --quiet
  fi
done
uv pip install --python "$python" --target site --require-hashes -r requirements.txt
uv venv --allow-existing --python "$python" build-venv
uv pip install --python build-venv/bin/python --require-hashes -r build.lock
archive=gtsam-4.3.0.tar.gz
if [[ ! -f "$archive" ]]; then
  curl -fsSL https://codeload.github.com/borglab/gtsam/tar.gz/refs/tags/4.3.0 -o "$archive"
fi
echo "4539afcd47d336864deb8399e0a9a5c283a638f605892c942ec46f3541fd4fb0  $archive" | shasum -a 256 -c -
tar -xzf "$archive"
prefix="$PWD/build-venv/lib/python3.12/site-packages/cmeel.prefix"
# Configure the wheel's matching release headers; link directly to its shipped library.
build-venv/bin/cmake -S gtsam-4.3.0 -B headers -DCMAKE_PREFIX_PATH="$prefix" \
  -DGTSAM_WITH_TBB=OFF -DGTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR=ON \
  -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_UNSTABLE=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_BUILD_PYTHON=OFF > configure.log
ext=$( "$python" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))' )
include=$( "$python" -c 'import sysconfig; print(sysconfig.get_path("include"))' )
lib=libgtsam.4.3.0.dylib
out="site/_gtsam_cstr$ext"
clang++ -O3 -DNDEBUG -std=c++17 -bundle -undefined dynamic_lookup \
  -I"$include" -Igtsam-4.3.0/wrap/pybind11/include -Igtsam-4.3.0/gtsam/3rdparty/Eigen \
  -I"$prefix/include" -Iheaders -Igtsam-4.3.0 "$source_dir/gtsam_cstr.cpp" \
  "site/gtsam/.dylibs/$lib" -o "$out"
install_name_tool -change "/DLC/gtsam/.dylibs/$lib" "@loader_path/gtsam/.dylibs/$lib" "$out"
codesign --force --sign - "$out"

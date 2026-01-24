#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
OPENCV_DIR_DEFAULT="/opt/opencv-4.13/lib/cmake/opencv4"
OPENCV_DIR="${OpenCV_DIR:-$OPENCV_DIR_DEFAULT}"
ONNXRUNTIME_ROOT="${ONNXRUNTIME_ROOT:-}"

rm -rf "${BUILD_DIR}"
if [[ -n "${ONNXRUNTIME_ROOT}" ]]; then
  cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DOpenCV_DIR="${OPENCV_DIR}" -DONNXRUNTIME_ROOT="${ONNXRUNTIME_ROOT}" -DCMAKE_BUILD_TYPE=Release
else
  cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DOpenCV_DIR="${OPENCV_DIR}" -DCMAKE_BUILD_TYPE=Release
fi
cmake --build "${BUILD_DIR}" -j
"${BUILD_DIR}/cat_feeder" --model "models/yolo26n.onnx" --width 1280 --height 720 --fps 30

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/output"
BIN_PATH="${BIN_PATH:-${ROOT_DIR}/build/preprocess_recorder}"

CAMERA_INDEX="${CAMERA_INDEX:-2}"
FPS="${FPS:-30}"
SIZE="${SIZE:-640x480}"
WIDTH="${WIDTH:-}"
HEIGHT="${HEIGHT:-}"
INPUT_SIZE="${INPUT_SIZE:-320}"
DURATION="${DURATION:-}"
STOP_KEY="${STOP_KEY:-q}"

print_usage() {
  cat <<'EOF'
Usage: scripts/record_video.sh [output_file.mp4] [--duration SECONDS]

Environment variables:
  BIN_PATH       Path to preprocess_recorder binary (default build/preprocess_recorder)
  CAMERA_INDEX   Camera index (default 2)
  FPS            Capture FPS (default 30)
  SIZE           Capture size, e.g. 640x480 (default 640x480)
  WIDTH          Capture width (overrides SIZE)
  HEIGHT         Capture height (overrides SIZE)
  INPUT_SIZE     Preprocess input size (default 320)
  DURATION       Duration in seconds (default: until stop key)
  STOP_KEY       Key to stop recording (default q, use CTRL+Q for ^Q)

Examples:
  scripts/record_video.sh
  scripts/record_video.sh output/cam2.avi --duration 20
  CAMERA_INDEX=2 FPS=25 SIZE=1280x720 scripts/record_video.sh
EOF
}

OUTPUT="${OUT_DIR}/preprocessed_cam${CAMERA_INDEX}_$(date +%Y%m%d_%H%M%S).avi"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  print_usage
  exit 0
fi

if [[ "${1:-}" == "--duration" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "Missing value for --duration" >&2
    exit 1
  fi
  DURATION="$2"
elif [[ -n "${1:-}" ]]; then
  OUTPUT="$1"
  shift
  if [[ "${1:-}" == "--duration" ]]; then
    if [[ -z "${2:-}" ]]; then
      echo "Missing value for --duration" >&2
      exit 1
    fi
    DURATION="$2"
  fi
fi

mkdir -p "${OUT_DIR}"

if [[ -z "${WIDTH}" || -z "${HEIGHT}" ]]; then
  if [[ "${SIZE}" =~ ^[0-9]+x[0-9]+$ ]]; then
    WIDTH="${SIZE%x*}"
    HEIGHT="${SIZE#*x}"
  else
    WIDTH="640"
    HEIGHT="480"
  fi
fi

if [[ ! -x "${BIN_PATH}" ]]; then
  echo "Binary not found: ${BIN_PATH}" >&2
  echo "Build it first (cmake --build build -j) or set BIN_PATH." >&2
  exit 1
fi

APP_ARGS=(
  --camera "${CAMERA_INDEX}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --fps "${FPS}"
  --input "${INPUT_SIZE}"
  --output "${OUTPUT}"
)

echo "Recording preprocessed video to ${OUTPUT}"
echo "Press '${STOP_KEY}' (or Ctrl+C) to stop."

"${BIN_PATH}" "${APP_ARGS[@]}" &
APP_PID=$!

cleanup() {
  stty sane 2>/dev/null || true
}
trap cleanup EXIT

stty -echo -icanon time 0 min 0
if [[ -n "${DURATION}" ]]; then
  (sleep "${DURATION}" && kill -INT "${APP_PID}" 2>/dev/null) &
fi

while kill -0 "${APP_PID}" 2>/dev/null; do
  if IFS= read -rsn1 key; then
    if [[ "${key}" == "${STOP_KEY}" ]]; then
      kill -INT "${APP_PID}" 2>/dev/null || true
      break
    fi
  fi
  sleep 0.05
done

wait "${APP_PID}"

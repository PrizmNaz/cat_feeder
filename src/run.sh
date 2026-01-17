#!/usr/bin/env bash

VIDEO_DEVICE=/dev/video0
IMAGE_NAME=vision:dev

docker build -t ${IMAGE_NAME} .

docker run --rm -it \
    --device ${VIDEO_DEVICE}:${VIDEO_DEVICE} \
    -e VIDEO_DEVICE=${VIDEO_DEVICE} \
    -p 8000:8000 \
    ${IMAGE_NAME}

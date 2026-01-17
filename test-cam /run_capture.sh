#!/usr/bin/env bash

VIDEO_DEVICE=/dev/video2
IMAGE_NAME=camtest:dev

xhost +local:docker

docker run --rm -it \
    --device ${VIDEO_DEVICE}:${VIDEO_DEVICE} \
    -e VIDEO_DEVICE=/dev/video0 \
    -e DISPLAY=${DISPLAY} \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    ${IMAGE_NAME}

xhost -local:docker
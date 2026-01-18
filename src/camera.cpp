#include "camera.h"
#include <sstream>

Camera::Camera(int index) : index_(index) {}

void Camera::request(int width, int height, double fps) {
    req_w_ = width;
    req_h_ = height;
    req_fps_ = fps;
}

bool Camera::open() {
    if (cap_.isOpened()) {
        cap_.release();
    }

    cap_.open(index_, cv::CAP_V4L2);
    if (!cap_.isOpened()) {
        cap_.open(index_, cv::CAP_ANY);
    }
    if (!cap_.isOpened()) return false;

    cap_.set(cv::CAP_PROP_FRAME_WIDTH,  req_w_);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, req_h_);
    cap_.set(cv::CAP_PROP_FPS,          req_fps_);

    return true;
}

bool Camera::read(cv::Mat& outFrame) {
    if (!cap_.isOpened()) {
        return false;
    }
    return cap_.read(outFrame);
}

std::string Camera::info() const {
    std::ostringstream ss;
    if (!cap_.isOpened()) {
        ss << "Camera: not opened";
        return ss.str();
    }

    const double w   = cap_.get(cv::CAP_PROP_FRAME_WIDTH);
    const double h   = cap_.get(cv::CAP_PROP_FRAME_HEIGHT);
    const double fps = cap_.get(cv::CAP_PROP_FPS);

    ss << "Camera Info: Index: " << index_
       << ", Width: " << (int)w
       << ", Height: " << (int)h
       << ", FPS: " << fps;

    return ss.str();
}

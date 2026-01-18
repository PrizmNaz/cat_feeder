#include "camera.h"

#include <iostream>
Camera::Camera(int index)
    : index_(index) {}

void Camera::request(int width, int height, double fps) {
    if (width > 0){
        req_w_ = width;
    }
    if (height > 0){
        req_h_ = height;
    }
    if (fps > 0.0){
        req_fps_ = fps;
    }
}

bool Camera::open() {
    cap_.open(index_);
    if (!cap_.isOpened()) {
        std::cerr << "Error: Could not open camera with index " << index_ << std::endl;
        return false;
    }
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, req_w_);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, req_h_);
    cap_.set(cv::CAP_PROP_FPS, req_fps_);
    return true;
}

bool Camera::read(cv::Mat& outFrame) {
    return cap_.read(outFrame);
}

std::string Camera::info() const {
    std::ostringstream ss;

    if (!cap_.isOpened()) {
        ss << "Camera not opened.";
        return ss.str();
    }
    const int width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    const double fps = cap_.get(cv::CAP_PROP_FPS);

    ss << "Camera Info: " <<
        "Index: " << std::to_string(index_)  <<
        ", Width: " << std::to_string(width) <<
        ", Height: " << std::to_string(height) <<
        ", FPS: " << std::to_string(fps);
        
    return ss.str();
}
#include "detector.h"

Detector::Detector(std::string onnxPath)
    : onnxPath_(std::move(onnxPath)) {}

bool Detector::load() {
    try {
        net_ = cv::dnn::readNetFromONNX(onnxPath_);
        if (net_.empty()) return false;

        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        return true;
    } catch (...) {
        return false;
    }
}

std::vector<cv::Mat> Detector::infer(const cv::Mat& blob) {
    std::vector<cv::Mat> outs;
    net_.setInput(blob);
    auto names = net_.getUnconnectedOutLayersNames();
    net_.forward(outs, names);
    return outs;
}

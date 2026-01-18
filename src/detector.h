#pragma once
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

class Detector {
public:
    explicit Detector(std::string onnxPath);

    bool load();
    std::vector<cv::Mat> infer(const cv::Mat& blob);

private:
    std::string onnxPath_;
    cv::dnn::Net net_;
};

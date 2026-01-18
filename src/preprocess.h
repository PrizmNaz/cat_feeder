
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

struct PreprocessResult {
    cv::Mat letterbox_bgr; // dst x dst
    cv::Mat blob;          // 1x3xdstxdst CV_32F NCHW
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
};

class Preprocessor {
public:
    Preprocessor(int dstSize = 320);
    PreprocessResult run(const cv::Mat& frame_bgr) const;

private:
    int dst_;
};

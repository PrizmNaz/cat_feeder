#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>


struct PreprocessResult {
    cv::Mat letterbox_bgr;  // 320x320 BGR (для дебага/сохранения)
    cv::Mat blob;           // 1x3x320x320 CV_32F NCHW

    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
};

class Preprocessor {
public:
    Preprocessor(int dstSize = 320, int stride = 32);

    PreprocessResult run(const cv::Mat& frame_bgr) const;

private:
    int dst_;      // целевой размер (dst_ x dst_)
    int stride_;   // для YOLO часто 32, чтобы размер был кратен stride
};

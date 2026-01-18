#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

#include "preprocessor.h"

struct Detection {
    cv::Rect2f box;
    float confidence = 0.0f;
    int class_id = -1;
};

struct PostprocessOptions {
    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;
};

class Postprocessor {
public:
    explicit Postprocessor(PostprocessOptions options);

    std::vector<Detection> run(
        const std::vector<cv::Mat>& outs,
        const PreprocessResult& prep,
        const cv::Size& orig_size
    ) const;

    void setDebug(bool enabled, int every_n = 30, const std::string& out_dir = "output");
    void maybeDebugFrame(const cv::Mat& frame, const std::vector<Detection>& dets);

private:
    PostprocessOptions options_;
    bool debug_enabled_ = false;
    int debug_every_n_ = 30;
    std::string debug_out_dir_ = "output";
    std::size_t frame_idx_ = 0;

    static cv::Mat to2D(const cv::Mat& out);
    static cv::Rect2f clipRect(const cv::Rect2f& r, const cv::Size& size);
};

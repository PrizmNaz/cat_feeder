#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "preprocessor.h"

struct Detection {
    cv::Rect2f box;
    float confidence = 0.0f;
    int class_id = -1;
};

struct PostprocessOptions {
    int num_classes = 0;          // 0 = auto-detect from output shape
    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;
    bool has_objectness = false;  // set true for YOLOv5-style outputs
    bool boxes_in_xywh = true;    // center-based boxes in network space
};

class Postprocessor {
public:
    explicit Postprocessor(PostprocessOptions options);

    std::vector<Detection> run(
        const std::vector<cv::Mat>& outs,
        const PreprocessResult& prep,
        const cv::Size& orig_size
    );

    void setDebug(bool enabled, int every_n = 30, const std::string& out_dir = "output");
    void maybeSaveDebug(const cv::Mat& frame, const std::vector<Detection>& dets);

private:
    PostprocessOptions options_;
    std::size_t frame_idx_ = 0;
    bool debug_enabled_ = false;
    int debug_every_n_ = 30;
    std::string debug_out_dir_ = "output";

    static cv::Mat to2D(const cv::Mat& out);
    static cv::Rect2f clipRect(const cv::Rect2f& r, const cv::Size& size);
    void decodeOutput(
        const cv::Mat& out2d,
        const PreprocessResult& prep,
        const cv::Size& orig_size,
        std::vector<Detection>& candidates
    ) const;
};

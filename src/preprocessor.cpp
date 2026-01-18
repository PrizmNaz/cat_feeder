#include "preprocessor.h"
#include <algorithm>
#include <cmath>

Preprocessor::Preprocessor(int dstSize, int stride)
    : dst_(dstSize), stride_(stride) {}

PreprocessResult Preprocessor::run(const cv::Mat& src) const {
    PreprocessResult res;
    if (src.empty()) return res;

    const int src_w = src.cols;
    const int src_h = src.rows;

    // scale so that image fits into dst_ x dst_ preserving aspect ratio
    const float r = std::min(static_cast<float>(dst_) / src_w,
                             static_cast<float>(dst_) / src_h);

    const int new_w = static_cast<int>(std::round(src_w * r));
    const int new_h = static_cast<int>(std::round(src_h * r));

    int pad_w = dst_ - new_w;
    int pad_h = dst_ - new_h;

    const int pad_left = pad_w / 2;
    const int pad_right = pad_w - pad_left;
    const int pad_top = pad_h / 2;
    const int pad_bottom = pad_h - pad_top;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat bordered;
    cv::copyMakeBorder(
        resized,
        bordered,
        pad_top, pad_bottom, pad_left, pad_right,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114)
    );

    // Ensure exact size
    if (bordered.cols != dst_ || bordered.rows != dst_) {
        cv::resize(bordered, bordered, cv::Size(dst_, dst_));
    }

    res.letterbox_bgr = bordered;
    res.scale = r;
    res.pad_x = pad_left;
    res.pad_y = pad_top;

    // Build blob: float32, NCHW, normalized 0..1, swapRB BGR->RGB
    res.blob = cv::dnn::blobFromImage(
        res.letterbox_bgr,
        1.0 / 255.0,
        cv::Size(dst_, dst_),
        cv::Scalar(0, 0, 0),
        true,   // swapRB
        false,  // crop
        CV_32F
    );

    return res;
}

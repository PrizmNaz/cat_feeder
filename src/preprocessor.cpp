
#include "preprocessor.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace {
    cv::Mat autoWhiteBalanceAndBrightness(const cv::Mat& src) {
        if (src.empty()) {
            return src;
        }

        cv::Mat input;
        if (src.type() == CV_8UC3) {
            input = src;
        } else {
            src.convertTo(input, CV_8UC3);
        }

        cv::Scalar mean_bgr = cv::mean(input);
        const double mean_gray = (mean_bgr[0] + mean_bgr[1] + mean_bgr[2]) / 3.0;

        double gains[3] = {1.0, 1.0, 1.0};
        for (int i = 0; i < 3; ++i) {
            if (mean_bgr[i] > 1e-6) {
                double g = mean_gray / mean_bgr[i];
                gains[i] = std::min(2.0, std::max(0.5, g));
            }
        }

        std::vector<cv::Mat> channels;
        cv::split(input, channels);
        for (int i = 0; i < 3; ++i) {
            channels[i].convertTo(channels[i], CV_8U, gains[i], 0.0);
        }
        cv::Mat balanced;
        cv::merge(channels, balanced);

        cv::Mat gray;
        cv::cvtColor(balanced, gray, cv::COLOR_BGR2GRAY);
        const double gray_mean = cv::mean(gray)[0];
        if (gray_mean > 1e-6) {
            const double target = 128.0;
            double alpha = target / gray_mean;
            alpha = std::min(1.5, std::max(0.5, alpha));
            cv::Mat bright;
            balanced.convertTo(bright, CV_8U, alpha, 0.0);
            return bright;
        }

        return balanced;
    }
}

Preprocessor::Preprocessor(int dstSize) : dst_(dstSize) {}

PreprocessResult Preprocessor::run(const cv::Mat& src) const {
    PreprocessResult res;
    if (src.empty() || dst_ <= 0) return res;

    const int src_w = src.cols;
    const int src_h = src.rows;
    if (src_w <= 0 || src_h <= 0) return res;

    cv::Mat corrected = autoWhiteBalanceAndBrightness(src);

    const float r = std::min((float)dst_ / src_w, (float)dst_ / src_h);
    const int new_w = std::max(1, (int)std::round(src_w * r));
    const int new_h = std::max(1, (int)std::round(src_h * r));

    const int pad_w = dst_ - new_w;
    const int pad_h = dst_ - new_h;

    const int pad_left = pad_w / 2;
    const int pad_right = pad_w - pad_left;
    const int pad_top = pad_h / 2;
    const int pad_bottom = pad_h - pad_top;

    cv::Mat resized;
    cv::resize(corrected, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat bordered;
    cv::copyMakeBorder(
        resized, bordered,
        pad_top, pad_bottom, pad_left, pad_right,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114)
    );

    if (bordered.cols != dst_ || bordered.rows != dst_) {
        cv::resize(bordered, bordered, cv::Size(dst_, dst_));
    }

    res.letterbox_bgr = bordered;
    res.scale = r;
    res.pad_x = pad_left;
    res.pad_y = pad_top;

    res.blob = cv::dnn::blobFromImage(
        res.letterbox_bgr,
        1.0 / 255.0,
        cv::Size(dst_, dst_),
        cv::Scalar(0,0,0),
        true,   // swapRB
        false,  // crop
        CV_32F
    );

    return res;
}

#include "postprocessor.h"

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <opencv2/dnn.hpp>

Postprocessor::Postprocessor(PostprocessOptions options)
    : options_(options) {}

cv::Mat Postprocessor::to2D(const cv::Mat& out) {
    if (out.dims <= 2) {
        return out;
    }

    cv::Mat cont = out;
    if (!out.isContinuous()) {
        cont = out.clone();
    }

    const int last_dim = out.size[out.dims - 1];
    const int rows = static_cast<int>(cont.total() / last_dim);
    cv::Mat out2d = cont.reshape(1, rows);

    if (out2d.rows < out2d.cols && out2d.rows <= 256) {
        out2d = out2d.t();
    }

    return out2d;
}

cv::Rect2f Postprocessor::clipRect(const cv::Rect2f& r, const cv::Size& size) {
    const float x0 = std::max(0.0f, r.x);
    const float y0 = std::max(0.0f, r.y);
    const float x1 = std::min(static_cast<float>(size.width - 1), r.x + r.width);
    const float y1 = std::min(static_cast<float>(size.height - 1), r.y + r.height);

    if (x1 <= x0 || y1 <= y0) {
        return cv::Rect2f();
    }

    return cv::Rect2f(x0, y0, x1 - x0, y1 - y0);
}

std::vector<Detection> Postprocessor::run(
    const std::vector<cv::Mat>& outs,
    const PreprocessResult& prep,
    const cv::Size& orig_size
) const {
    std::vector<Detection> candidates;

    for (const auto& out : outs) {
        cv::Mat out2d = to2D(out);
        if (out2d.empty() || out2d.cols < 6) {
            continue;
        }

        for (int i = 0; i < out2d.rows; ++i) {
            const float* data = out2d.ptr<float>(i);

            float x1 = data[0];
            float y1 = data[1];
            float x2 = data[2];
            float y2 = data[3];
            const float conf = data[4];
            const int class_id = static_cast<int>(data[5]);

            if (conf < options_.conf_threshold) {
                continue;
            }

            cv::Rect2f box(x1, y1, x2 - x1, y2 - y1);

            if (prep.scale > 0.0f) {
                const float bx0 = (box.x - prep.pad_x) / prep.scale;
                const float by0 = (box.y - prep.pad_y) / prep.scale;
                const float bx1 = (box.x + box.width - prep.pad_x) / prep.scale;
                const float by1 = (box.y + box.height - prep.pad_y) / prep.scale;
                box = cv::Rect2f(bx0, by0, bx1 - bx0, by1 - by0);
            }

            box = clipRect(box, orig_size);
            if (box.width <= 0.0f || box.height <= 0.0f) {
                continue;
            }

            candidates.push_back({box, conf, class_id});
        }
    }

    if (candidates.empty()) {
        return candidates;
    }

    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    boxes.reserve(candidates.size());
    scores.reserve(candidates.size());
    for (const auto& det : candidates) {
        boxes.emplace_back(det.box.x, det.box.y, det.box.width, det.box.height);
        scores.push_back(det.confidence);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, options_.conf_threshold, options_.nms_threshold, indices);

    std::vector<Detection> result;
    result.reserve(indices.size());
    for (int idx : indices) {
        result.push_back(candidates[idx]);
    }

    return result;
}

void Postprocessor::setDebug(bool enabled, int every_n, const std::string& out_dir) {
    debug_enabled_ = enabled;
    debug_every_n_ = std::max(1, every_n);
    debug_out_dir_ = out_dir;
}

void Postprocessor::maybeDebugFrame(const cv::Mat& frame, const std::vector<Detection>& dets) {
    ++frame_idx_;
    if (!debug_enabled_) return;
    if (frame.empty()) return;
    if ((frame_idx_ % static_cast<std::size_t>(debug_every_n_)) != 0) return;

    std::cout << "detections: " << dets.size() << "\n";
    for (const auto& det : dets) {
        std::cout << "  class=" << det.class_id
                  << " conf=" << det.confidence
                  << " box=[" << det.box.x << "," << det.box.y
                  << "," << det.box.width << "," << det.box.height << "]\n";
    }

    std::filesystem::create_directories(debug_out_dir_);

    cv::Mat vis = frame.clone();
    drawDetections(vis, dets);

    std::ostringstream path;
    path << debug_out_dir_ << "/frame_" << std::setw(6) << std::setfill('0') << frame_idx_ << ".jpg";
    cv::imwrite(path.str(), vis);
}

void Postprocessor::drawDetections(cv::Mat& frame, const std::vector<Detection>& dets) const {
    for (const auto& det : dets) {
        cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);

        std::ostringstream label;
        label << det.class_id << " " << std::fixed << std::setprecision(2) << det.confidence;
        cv::putText(
            frame, label.str(),
            cv::Point(static_cast<int>(det.box.x), std::max(0, static_cast<int>(det.box.y) - 4)),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1
        );
    }
}

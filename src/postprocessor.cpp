#include "postprocessor.h"

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <sstream>

#include <opencv2/dnn.hpp>

Postprocessor::Postprocessor(PostprocessOptions options)
    : options_(options) {}

cv::Mat Postprocessor::to2D(const cv::Mat& out) {
    if (out.dims <= 2) {
        return out;
    }

    const int last_dim = out.size[out.dims - 1];
    const int rows = static_cast<int>(out.total() / last_dim);
    cv::Mat out2d = out.reshape(1, rows);

    // Heuristic: if attributes axis is the first (small) dimension, transpose.
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

void Postprocessor::decodeOutput(
    const cv::Mat& out2d,
    const PreprocessResult& prep,
    const cv::Size& orig_size,
    std::vector<Detection>& candidates
) const {
    if (out2d.empty()) return;
    if (out2d.cols < 5) return;

    int num_classes = options_.num_classes;
    bool has_obj = options_.has_objectness;

    if (num_classes > 0) {
        if (out2d.cols == 5 + num_classes) {
            has_obj = true;
        } else if (out2d.cols == 4 + num_classes) {
            has_obj = false;
        }
    } else {
        // Auto-detect class count using common YOLO layouts.
        if (out2d.cols >= 6) {
            num_classes = out2d.cols - (options_.has_objectness ? 5 : 4);
        }
    }

    const int cls_start = has_obj ? 5 : 4;
    const int cls_count = out2d.cols - cls_start;
    if (cls_count <= 0) return;

    const float input_w = static_cast<float>(prep.letterbox_bgr.cols);
    const float input_h = static_cast<float>(prep.letterbox_bgr.rows);

    for (int i = 0; i < out2d.rows; ++i) {
        const float* data = out2d.ptr<float>(i);

        float x = data[0];
        float y = data[1];
        float w = data[2];
        float h = data[3];

        const float max_coord = std::max(std::max(x, y), std::max(w, h));
        if (max_coord <= 1.5f) {
            x *= input_w;
            y *= input_h;
            w *= input_w;
            h *= input_h;
        }

        int best_class = -1;
        float best_score = 0.0f;
        for (int c = 0; c < cls_count; ++c) {
            const float score = data[cls_start + c];
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }

        float confidence = best_score;
        if (has_obj) {
            confidence *= data[4];
        }

        if (confidence < options_.conf_threshold) {
            continue;
        }

        cv::Rect2f box;
        if (options_.boxes_in_xywh) {
            const float x0 = x - w * 0.5f;
            const float y0 = y - h * 0.5f;
            box = cv::Rect2f(x0, y0, w, h);
        } else {
            box = cv::Rect2f(x, y, w - x, h - y);
        }

        // Map from letterboxed input to original frame.
        if (prep.scale > 0.0f) {
            const float x0 = (box.x - prep.pad_x) / prep.scale;
            const float y0 = (box.y - prep.pad_y) / prep.scale;
            const float x1 = (box.x + box.width - prep.pad_x) / prep.scale;
            const float y1 = (box.y + box.height - prep.pad_y) / prep.scale;
            box = cv::Rect2f(x0, y0, x1 - x0, y1 - y0);
        }

        box = clipRect(box, orig_size);
        if (box.width <= 0.0f || box.height <= 0.0f) {
            continue;
        }

        candidates.push_back({box, confidence, best_class});
    }
}

std::vector<Detection> Postprocessor::run(
    const std::vector<cv::Mat>& outs,
    const PreprocessResult& prep,
    const cv::Size& orig_size
) {
    std::vector<Detection> candidates;

    for (const auto& out : outs) {
        cv::Mat out2d = to2D(out);
        decodeOutput(out2d, prep, orig_size, candidates);
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

void Postprocessor::maybeSaveDebug(const cv::Mat& frame, const std::vector<Detection>& dets) {
    ++frame_idx_;
    if (!debug_enabled_) return;
    if (frame.empty()) return;
    if ((frame_idx_ % static_cast<std::size_t>(debug_every_n_)) != 0) return;

    std::filesystem::create_directories(debug_out_dir_);

    cv::Mat vis = frame.clone();
    for (const auto& det : dets) {
        cv::rectangle(vis, det.box, cv::Scalar(0, 255, 0), 2);

        std::ostringstream label;
        label << det.class_id << " " << std::fixed << std::setprecision(2) << det.confidence;
        cv::putText(
            vis, label.str(),
            cv::Point(static_cast<int>(det.box.x), std::max(0, static_cast<int>(det.box.y) - 4)),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1
        );
    }

    std::ostringstream path;
    path << debug_out_dir_ << "/frame_" << std::setw(6) << std::setfill('0') << frame_idx_ << ".jpg";
    cv::imwrite(path.str(), vis);
}

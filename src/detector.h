#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

class Detector {
public:
    explicit Detector(std::string modelPath);

    bool load();
    std::vector<cv::Mat> infer(const cv::Mat& blob);

private:
    std::string model_path_;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "cat_feeder"};
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_name_ptrs_;
    std::vector<const char*> output_name_ptrs_;
    bool loaded_ = false;
};

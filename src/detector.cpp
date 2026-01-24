#include "detector.h"

#include <cstring>
#include <iostream>

Detector::Detector(std::string modelPath)
    : model_path_(std::move(modelPath)) {}

bool Detector::load() {
    try {
        // Basic session tuning for CPU inference.
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_ = std::make_unique<Ort::Session>(env_, model_path_.c_str(), session_options_);

        // Cache input/output names for faster Run() calls.
        const std::size_t num_inputs = session_->GetInputCount();
        input_names_.clear();
        input_name_ptrs_.clear();
        input_names_.reserve(num_inputs);
        input_name_ptrs_.reserve(num_inputs);
        for (std::size_t i = 0; i < num_inputs; ++i) {
            Ort::AllocatedStringPtr name = session_->GetInputNameAllocated(i, allocator_);
            input_names_.emplace_back(name.get());
        }
        for (const auto& name : input_names_) {
            input_name_ptrs_.push_back(name.c_str());
        }

        const std::size_t num_outputs = session_->GetOutputCount();
        output_names_.clear();
        output_name_ptrs_.clear();
        output_names_.reserve(num_outputs);
        output_name_ptrs_.reserve(num_outputs);
        for (std::size_t i = 0; i < num_outputs; ++i) {
            Ort::AllocatedStringPtr name = session_->GetOutputNameAllocated(i, allocator_);
            output_names_.emplace_back(name.get());
        }
        for (const auto& name : output_names_) {
            output_name_ptrs_.push_back(name.c_str());
        }

        loaded_ = true;
        return true;
    } catch (const Ort::Exception& ex) {
        std::cerr << "Detector load failed: " << ex.what() << "\n";
        return false;
    } catch (...) {
        return false;
    }
}

static cv::Mat tensorToMat(const Ort::Value& tensor) {
    if (!tensor.IsTensor()) {
        return cv::Mat();
    }

    auto info = tensor.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return cv::Mat();
    }

    std::vector<int64_t> shape = info.GetShape();
    if (shape.empty()) {
        return cv::Mat();
    }

    std::vector<int> sizes;
    sizes.reserve(shape.size());
    for (int64_t dim : shape) {
        if (dim <= 0) {
            return cv::Mat();
        }
        sizes.push_back(static_cast<int>(dim));
    }

    cv::Mat out(static_cast<int>(sizes.size()), sizes.data(), CV_32F);
    const std::size_t bytes = info.GetElementCount() * sizeof(float);
    std::memcpy(out.data, tensor.GetTensorData<float>(), bytes);
    return out;
}

std::vector<cv::Mat> Detector::infer(const cv::Mat& blob) {
    std::vector<cv::Mat> outs;
    if (!loaded_ || !session_ || blob.empty()) {
        return outs;
    }
    if (blob.dims != 4 || blob.type() != CV_32F) {
        return outs;
    }

    // Ensure contiguous storage for zero-copy input tensor.
    cv::Mat contiguous = blob.isContinuous() ? blob : blob.clone();
    const int64_t batch = contiguous.size[0];
    const int64_t channels = contiguous.size[1];
    const int64_t height = contiguous.size[2];
    const int64_t width = contiguous.size[3];
    if (batch != 1 || channels != 3) {
        return outs;
    }

    const std::vector<int64_t> input_shape = {batch, channels, height, width};
    const std::size_t input_size = static_cast<std::size_t>(batch * channels * height * width);
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // Create a CPU tensor that views the blob memory (no extra copy).
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        const_cast<float*>(contiguous.ptr<float>()),
        input_size,
        input_shape.data(),
        input_shape.size()
    );

    try {
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_name_ptrs_.data(),
            &input_tensor,
            1,
            output_name_ptrs_.data(),
            output_name_ptrs_.size()
        );

        // Postprocessor expects each output as a cv::Mat with original tensor dims.
        outs.reserve(output_tensors.size());
        for (const auto& output : output_tensors) {
            cv::Mat out = tensorToMat(output);
            if (!out.empty()) {
                outs.push_back(out);
            }
        }
    } catch (const Ort::Exception& ex) {
        std::cerr << "Detector inference failed: " << ex.what() << "\n";
    }

    return outs;
}

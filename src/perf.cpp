#include "perf.h"

void Perf::onFrame(double captureMs, double preprocessMs, double totalMs) {
    captureMs = (captureMs > 0.0) ? captureMs : 0.0;
    preprocessMs = (preprocessMs > 0.0) ? preprocessMs : 0.0;
    totalMs = (totalMs > 0.0) ? totalMs : 0.0;
    frames_++;
    
    sum_capture_ms_ += captureMs;
    sum_preprocess_ms_ += preprocessMs;
    sum_total_ms_ += totalMs;
}

bool Perf::shouldReport() const {
    auto now_ = clock::now();
    auto ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(now_ - last_report_).count();
    return ms_ >= 1000;
}

void Perf::markReported() {
    last_report_ = clock::now();
}

double Perf::avgFps() const {
    auto now_ = clock::now();
    auto total_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(now_ - start_).count();
    double total_s_ = total_ms_ / 1000.0;
    return (total_s_ > 0.0) ?frames_/total_s_ : 0.0;
}

double Perf::avgCaptureMs() const{
    return (frames_ > 0) ? sum_capture_ms_ / frames_ : 0.0;
}

double Perf::avgPreprocessMs() const{
    return (frames_ > 0) ? sum_preprocess_ms_ / frames_ : 0.0;
}

double Perf::avgTotalMs() const{
    return (frames_ > 0) ? sum_total_ms_ / frames_ : 0.0;
}
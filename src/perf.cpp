
#include "perf.h"

void Perf::onFrame(double captureMs, double preprocessMs, double inferMs, double totalMs) {
    frames_++;
    sum_capture_ms_ += captureMs;
    sum_preprocess_ms_ += preprocessMs;
    sum_infer_ms_ += inferMs;
    sum_total_ms_ += totalMs;
}

bool Perf::shouldReport() const {
    auto now = clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_report_).count();
    return ms >= 1000;
}

void Perf::markReported() {
    last_report_ = clock::now();
}

double Perf::avgFps() const {
    auto now = clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_).count();
    double total_s = total_ms / 1000.0;
    return (total_s > 0.0) ? (frames_ / total_s) : 0.0;
}

double Perf::avgCaptureMs() const { return frames_ ? sum_capture_ms_ / frames_ : 0.0; }
double Perf::avgPreprocessMs() const { return frames_ ? sum_preprocess_ms_ / frames_ : 0.0; }
double Perf::avgInferMs() const { return frames_ ? sum_infer_ms_ / frames_ : 0.0; }
double Perf::avgTotalMs() const { return frames_ ? sum_total_ms_ / frames_ : 0.0; }

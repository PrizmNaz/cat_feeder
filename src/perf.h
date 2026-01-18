#pragma once
#include <chrono>
#include <cstddef>


class Perf {
    public:
        using clock = std::chrono::steady_clock;

        void onFrame(double captureMs, double preprocessMs, double totalMs);
        bool shouldReport() const;
        void markReported();

        std::size_t frames() const { return frames_; };

        double avgFps() const;
        double avgCaptureMs() const;
        double avgPreprocessMs() const;
        double avgTotalMs() const;

    private:
        clock::time_point start_ = clock::now();
        clock::time_point last_report_ = start_;
        std::size_t frames_ = 0;

        double sum_capture_ms_ = 0.0;
        double sum_preprocess_ms_ = 0.0;
        double sum_total_ms_ = 0.0;

};
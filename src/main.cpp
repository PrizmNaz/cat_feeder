#include <atomic>
#include <csignal>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

#include "camera.h"
#include "perf.h"

namespace {
    std::atomic<bool> g_running{true};
    void on_sigint(int) { g_running = false; }
}

int main() {
    std::signal(SIGINT, on_sigint);

    Camera cam(2);
    cam.request(640, 480, 30);

    if (!cam.open()) {
        std::cerr << "ERROR: cannot open camera\n";
        return 1;
    }

    std::cout << cam.info() << "\n";
    std::cout << "Capturing... Press Ctrl+C to stop.\n";

    Perf perf;
    cv::Mat frame;

    while (g_running) {
        auto t_total0 = Perf::clock::now();

        // --- CAPTURE ---
        auto t_cap0 = Perf::clock::now();
        if (!cam.read(frame) || frame.empty()) {
            std::cerr << "WARNING: failed to read frame\n";
            continue;
        }
        auto t_cap1 = Perf::clock::now();

        // --- PREPROCESS (имитация будущего YOLO-пайплайна) ---
        auto t_pre0 = Perf::clock::now();

        // 1) resize до "модельного" размера (например 320x320)
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(320, 320), 0, 0, cv::INTER_LINEAR);

        // 2) BGR -> RGB (часто модели ждут RGB)
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        // 3) небольшая нагрузка (blur), чтобы увидеть влияние
        cv::Mat blurred;
        cv::GaussianBlur(rgb, blurred, cv::Size(5, 5), 0);

        auto t_pre1 = Perf::clock::now();

        auto t_total1 = Perf::clock::now();

        double cap_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_cap1 - t_cap0).count() / 1000.0;
        double pre_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_pre1 - t_pre0).count() / 1000.0;
        double total_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_total1 - t_total0).count() / 1000.0;

        perf.onFrame(cap_ms, pre_ms, total_ms);

        if (perf.shouldReport()) {
            std::cout
                << "Frames: " << perf.frames()
                << " | Avg FPS: " << perf.avgFps()
                << " | capture(ms): " << perf.avgCaptureMs()
                << " | preprocess(ms): " << perf.avgPreprocessMs()
                << " | total(ms): " << perf.avgTotalMs()
                << "\n";
            perf.markReported();
        }
    }

    std::cout << "Stopped.\n";
    return 0;
}

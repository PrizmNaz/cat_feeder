#include <atomic>
#include <csignal>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "camera.h"
#include "preprocess.h"
#include "perf.h"
#include "detector.h"

namespace {
    std::atomic<bool> g_running{true};
    void on_sigint(int) { g_running = false; }
}

int main() {
    std::signal(SIGINT, on_sigint);

    // --- Camera ---
    Camera cam(2);                 // поменяй индекс, если нужно
    cam.request(640, 480, 30);

    if (!cam.open()) {
        std::cerr << "ERROR: cannot open camera\n";
        return 1;
    }

    std::cout << cam.info() << "\n";

    // --- Preprocess (YOLO input size) ---
    Preprocessor prep(320);        // под твой экспорт (320)

    // --- Detector ---
    Detector det("models/yolo26n.onnx"); // имя файла поправь при необходимости
    if (!det.load()) {
        std::cerr << "ERROR: cannot load ONNX model\n";
        return 1;
    }

    std::cout << "Detector loaded\n";

    // --- Runtime ---
    Perf perf;
    cv::Mat frame;

    while (g_running) {
        auto t_total0 = Perf::clock::now();

        // CAPTURE
        auto t_cap0 = Perf::clock::now();
        if (!cam.read(frame) || frame.empty()) {
            continue;
        }
        auto t_cap1 = Perf::clock::now();

        // PREPROCESS
        auto t_pre0 = Perf::clock::now();
        PreprocessResult pr = prep.run(frame);
        auto t_pre1 = Perf::clock::now();

        // INFERENCE
        auto t_inf0 = Perf::clock::now();
        std::vector<cv::Mat> outs = det.infer(pr.blob);
        auto t_inf1 = Perf::clock::now();

        auto t_total1 = Perf::clock::now();

        double cap_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_cap1 - t_cap0).count() / 1000.0;
        double pre_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_pre1 - t_pre0).count() / 1000.0;
        double inf_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_inf1 - t_inf0).count() / 1000.0;
        double total_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_total1 - t_total0).count() / 1000.0;

        perf.onFrame(cap_ms, pre_ms, inf_ms, total_ms);

        if (perf.shouldReport()) {
            std::cout
                << "FPS: " << perf.avgFps()
                << " | cap(ms): " << perf.avgCaptureMs()
                << " | pre(ms): " << perf.avgPreprocessMs()
                << " | infer(ms): " << perf.avgInferMs()
                << " | total(ms): " << perf.avgTotalMs()
                << "\n";
            perf.markReported();
        }
    }

    std::cout << "Stopped\n";
    return 0;
}

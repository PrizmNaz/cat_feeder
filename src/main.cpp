#include <atomic>
#include <csignal>
#include <iostream>
#include <chrono>

#include "camera.h"
#include "perf.h"
#include "preprocessor.h"

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

    Preprocessor prep(320, 32);

    bool printed_blob_info = false;



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

        PreprocessResult pr = prep.run(frame);
        
        if (!printed_blob_info) {
            const cv::Mat& blob = pr.blob;
            std::cout << "Blob dims: " << blob.dims << "\n";
            if (blob.dims == 4) {
                std::cout << "Blob shape: ["
                        << blob.size[0] << ", "
                        << blob.size[1] << ", "
                        << blob.size[2] << ", "
                        << blob.size[3] << "]\n";
            }
            std::cout << "Blob type: " << blob.type() << " (expect CV_32F)\n";
            std::cout << "Letterbox: " << pr.letterbox_bgr.cols << "x" << pr.letterbox_bgr.rows
                    << " | scale=" << pr.scale << " pad_x=" << pr.pad_x << " pad_y=" << pr.pad_y << "\n";
            printed_blob_info = true;
        }


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

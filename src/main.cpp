#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "camera.h"
#include "preprocessor.h"
#include "perf.h"
#include "detector.h"
#include "postprocessor.h"
#include "web_server.h"

namespace {
    std::atomic<bool> g_running{true};
    void on_sigint(int) { g_running = false; }

    struct AppConfig {
        int camera_index = 2;
        int width = 640;
        int height = 480;
        double fps = 30.0;
        int input_size = 640;
        std::string model_path = "models/yolo26n.onnx";
        float conf_threshold = 0.25f;
        float nms_threshold = 0.45f;
        int port = 8080;
        int threads = 0;
        bool enable_web = true;
    };

    void printUsage(const char* exe) {
        std::cout
            << "Usage: " << exe << " [options]\n"
            << "Options:\n"
            << "  --camera <index>       Camera index (default 0)\n"
            << "  --width <px>           Capture width (default 640)\n"
            << "  --height <px>          Capture height (default 480)\n"
            << "  --fps <num>            Capture FPS (default 30)\n"
            << "  --input <px>           Model input size (default 320)\n"
            << "  --model <path>         ONNX model path\n"
            << "  --conf <0..1>          Confidence threshold (default 0.25)\n"
            << "  --nms <0..1>           NMS threshold (default 0.45)\n"
            << "  --port <num>           Web server port (default 8080)\n"
            << "  --threads <num>        OpenCV threads (default auto)\n"
            << "  --no-web               Disable web server\n"
            << "  --help                 Show this help\n";
    }

    bool parseInt(const char* text, int* out) {
        if (!text || !out) return false;
        char* end = nullptr;
        errno = 0;
        long value = std::strtol(text, &end, 10);
        if (errno != 0 || end == text || *end != '\0') return false;
        if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) return false;
        *out = static_cast<int>(value);
        return true;
    }

    bool parseDouble(const char* text, double* out) {
        if (!text || !out) return false;
        char* end = nullptr;
        errno = 0;
        double value = std::strtod(text, &end);
        if (errno != 0 || end == text || *end != '\0') return false;
        *out = value;
        return true;
    }

    bool parseFloat(const char* text, float* out) {
        if (!text || !out) return false;
        char* end = nullptr;
        errno = 0;
        float value = std::strtof(text, &end);
        if (errno != 0 || end == text || *end != '\0') return false;
        *out = value;
        return true;
    }

    int defaultThreads() {
        unsigned int hw = std::thread::hardware_concurrency();
        return hw > 0 ? static_cast<int>(hw) : 2;
    }

    bool parseArgs(int argc, char** argv, AppConfig* cfg, bool* show_help) {
        if (!cfg || !show_help) return false;
        *show_help = false;
        for (int i = 1; i < argc; ++i) {
            const char* arg = argv[i];
            if (std::strcmp(arg, "--help") == 0) {
                *show_help = true;
                return true;
            }
            if (std::strcmp(arg, "--no-web") == 0) {
                cfg->enable_web = false;
                continue;
            }
            if (i + 1 >= argc) {
                std::cerr << "Missing value for argument: " << arg << "\n";
                return false;
            }
            const char* val = argv[++i];
            if (std::strcmp(arg, "--camera") == 0) {
                if (!parseInt(val, &cfg->camera_index)) return false;
            } else if (std::strcmp(arg, "--width") == 0) {
                if (!parseInt(val, &cfg->width)) return false;
            } else if (std::strcmp(arg, "--height") == 0) {
                if (!parseInt(val, &cfg->height)) return false;
            } else if (std::strcmp(arg, "--fps") == 0) {
                if (!parseDouble(val, &cfg->fps)) return false;
            } else if (std::strcmp(arg, "--input") == 0) {
                if (!parseInt(val, &cfg->input_size)) return false;
            } else if (std::strcmp(arg, "--model") == 0) {
                cfg->model_path = val;
            } else if (std::strcmp(arg, "--conf") == 0) {
                if (!parseFloat(val, &cfg->conf_threshold)) return false;
            } else if (std::strcmp(arg, "--nms") == 0) {
                if (!parseFloat(val, &cfg->nms_threshold)) return false;
            } else if (std::strcmp(arg, "--port") == 0) {
                if (!parseInt(val, &cfg->port)) return false;
            } else if (std::strcmp(arg, "--threads") == 0) {
                if (!parseInt(val, &cfg->threads)) return false;
            } else {
                std::cerr << "Unknown argument: " << arg << "\n";
                return false;
            }
        }

        if (cfg->width <= 0 || cfg->height <= 0 || cfg->input_size <= 0) {
            std::cerr << "Invalid image size\n";
            return false;
        }
        if (cfg->fps <= 0.0) {
            std::cerr << "Invalid FPS\n";
            return false;
        }
        if (cfg->conf_threshold < 0.0f || cfg->conf_threshold > 1.0f) {
            std::cerr << "Invalid confidence threshold\n";
            return false;
        }
        if (cfg->nms_threshold < 0.0f || cfg->nms_threshold > 1.0f) {
            std::cerr << "Invalid NMS threshold\n";
            return false;
        }
        if (cfg->threads < 0) {
            std::cerr << "Invalid threads count\n";
            return false;
        }
        if (cfg->threads == 0) {
            cfg->threads = defaultThreads();
        }
        return true;
    }
}

int main(int argc, char** argv) {
    std::signal(SIGINT, on_sigint);

    AppConfig cfg;
    bool show_help = false;
    if (!parseArgs(argc, argv, &cfg, &show_help)) {
        std::cerr << "Use --help for usage.\n";
        return 1;
    }
    if (show_help) {
        printUsage(argv[0]);
        return 0;
    }

    cv::setUseOptimized(true);
    if (cfg.threads > 0) {
        cv::setNumThreads(cfg.threads);
    }

    // --- Camera ---
    Camera cam(cfg.camera_index);
    cam.request(cfg.width, cfg.height, cfg.fps);

    if (!cam.open()) {
        std::cerr << "ERROR: cannot open camera\n";
        return 1;
    }

    std::cout << cam.info() << "\n";

    // --- Preprocess (YOLO input size) ---
    Preprocessor prep(cfg.input_size);

    // --- Detector ---
    Detector det(cfg.model_path);
    if (!det.load()) {
        std::cerr << "ERROR: cannot load ONNX model\n";
        return 1;
    }

    std::cout << "Detector loaded\n";

    // --- Runtime ---
    Perf perf;
    PostprocessOptions post_opts;
    post_opts.conf_threshold = cfg.conf_threshold;
    post_opts.nms_threshold = cfg.nms_threshold;
    Postprocessor post(post_opts);
    // post.setDebug(true, 30, "output");

    WebServer server(cfg.port);
    if (cfg.enable_web) {
        if (server.start()) {
            std::cout << "Web UI: http://localhost:" << cfg.port << "\n";
        } else {
            std::cerr << "ERROR: cannot start web server\n";
        }
    }

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
        if (pr.blob.empty()) {
            continue;
        }

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

        std::vector<Detection> dets = post.run(outs, pr, frame.size());
        post.maybeDebugFrame(frame, dets);

        post.drawDetections(frame, dets);
        if (cfg.enable_web) {
            std::ostringstream stats;
            stats << std::fixed << std::setprecision(2)
                  << "fps=" << perf.avgFps()
                  << " cap_ms=" << perf.avgCaptureMs()
                  << " pre_ms=" << perf.avgPreprocessMs()
                  << " infer_ms=" << perf.avgInferMs()
                  << " total_ms=" << perf.avgTotalMs();
            server.updateFrame(frame, stats.str());
        }

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

#include <atomic>
#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <thread>

#include <opencv2/opencv.hpp>

#include "camera.h"
#include "preprocessor.h"

namespace {
    std::atomic<bool> g_running{true};
    void on_sigint(int) { g_running = false; }

    struct AppConfig {
        int camera_index = 2;
        int width = 640;
        int height = 480;
        double fps = 30.0;
        int input_size = 320;
        std::string output_path;
        int threads = 0;
    };

    void printUsage(const char* exe) {
        std::cout
            << "Usage: " << exe << " --output <path> [options]\n"
            << "Options:\n"
            << "  --camera <index>   Camera index (default 2)\n"
            << "  --width <px>       Capture width (default 640)\n"
            << "  --height <px>      Capture height (default 480)\n"
            << "  --fps <num>        Capture FPS (default 30)\n"
            << "  --input <px>       Preprocess input size (default 320)\n"
            << "  --threads <num>    OpenCV threads (default auto)\n"
            << "  --help             Show this help\n";
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
            } else if (std::strcmp(arg, "--threads") == 0) {
                if (!parseInt(val, &cfg->threads)) return false;
            } else if (std::strcmp(arg, "--output") == 0) {
                cfg->output_path = val;
            } else {
                std::cerr << "Unknown argument: " << arg << "\n";
                return false;
            }
        }

        if (cfg->output_path.empty()) {
            std::cerr << "Missing --output <path>\n";
            return false;
        }
        if (cfg->width <= 0 || cfg->height <= 0 || cfg->input_size <= 0) {
            std::cerr << "Invalid image size\n";
            return false;
        }
        if (cfg->fps <= 0.0) {
            std::cerr << "Invalid FPS\n";
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

    int fourccForPath(const std::string& path) {
        std::filesystem::path p(path);
        std::string ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (ext == ".avi") {
            return cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        }
        if (ext == ".mp4" || ext == ".m4v") {
            return cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        }
        return cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    }

    bool openWriter(cv::VideoWriter* writer, const std::string& path, double fps, const cv::Size& size) {
        if (!writer) return false;
        std::filesystem::path p(path);
        if (p.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(p.parent_path(), ec);
        }
        int fourcc = fourccForPath(path);
        if (writer->open(path, cv::CAP_FFMPEG, fourcc, fps, size, true)) {
            return true;
        }
        return writer->open(path, cv::CAP_ANY, fourcc, fps, size, true);
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

    Camera cam(cfg.camera_index);
    cam.request(cfg.width, cfg.height, cfg.fps);
    if (!cam.open()) {
        std::cerr << "ERROR: cannot open camera\n";
        return 1;
    }

    Preprocessor prep(cfg.input_size);

    cv::VideoWriter writer;
    std::string output_path = cfg.output_path;
    bool writer_opened = false;
    std::size_t frames_written = 0;

    cv::Mat frame;
    while (g_running) {
        if (!cam.read(frame) || frame.empty()) {
            continue;
        }

        PreprocessResult pr = prep.run(frame);
        if (pr.letterbox_bgr.empty()) {
            continue;
        }

        if (!writer_opened) {
            if (!openWriter(&writer, output_path, cfg.fps, pr.letterbox_bgr.size())) {
                std::filesystem::path p(output_path);
                std::filesystem::path fallback = p;
                fallback.replace_extension(".avi");
                std::cerr << "WARN: cannot open writer for " << output_path
                          << ", trying MJPG AVI: " << fallback.string() << "\n";
                output_path = fallback.string();
                if (!openWriter(&writer, output_path, cfg.fps, pr.letterbox_bgr.size())) {
                    std::cerr << "ERROR: cannot open output writer\n";
                    return 1;
                }
            }
            writer_opened = true;
            std::cout << "Recording to " << output_path << "\n";
        }

        writer.write(pr.letterbox_bgr);
        ++frames_written;
    }

    if (writer_opened) {
        writer.release();
    }
    if (frames_written == 0) {
        std::cerr << "WARN: no frames were written\n";
    }
    std::cout << "Stopped\n";
    return 0;
}

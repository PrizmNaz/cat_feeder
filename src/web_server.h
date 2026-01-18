#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

class WebServer {
public:
    explicit WebServer(int port = 8080);
    ~WebServer();

    bool start();
    void stop();

    void updateFrame(const cv::Mat& bgr, const std::string& stats_json);

private:
    void acceptLoop();
    void handleClient(int client_fd, const std::shared_ptr<std::atomic<bool>>& done);
    void serveIndex(int client_fd);
    void serveStats(int client_fd);
    void serveStream(int client_fd);

    static bool sendAll(int fd, const void* data, std::size_t size);

    int port_ = 8080;
    int listen_fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread server_thread_;

    std::mutex data_mutex_;
    std::condition_variable data_cv_;
    std::vector<uchar> latest_jpeg_;
    std::string stats_json_;
    std::uint64_t frame_id_ = 0;

    struct ClientWorker {
        std::thread thread;
        std::shared_ptr<std::atomic<bool>> done;
    };
    std::mutex clients_mutex_;
    std::vector<ClientWorker> client_workers_;
};

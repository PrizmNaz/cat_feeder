#include "web_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <cerrno>
#include <iostream>
#include <sstream>

WebServer::WebServer(int port) : port_(port) {}

WebServer::~WebServer() {
    stop();
}

bool WebServer::start() {
    if (running_.load()) {
        return true;
    }

    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        std::cerr << "WebServer: socket() failed\n";
        return false;
    }

    int opt = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port_));

    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "WebServer: bind() failed\n";
        ::close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }

    if (::listen(listen_fd_, 8) < 0) {
        std::cerr << "WebServer: listen() failed\n";
        ::close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }

    running_.store(true);
    server_thread_ = std::thread(&WebServer::acceptLoop, this);
    return true;
}

void WebServer::stop() {
    if (!running_.exchange(false)) {
        return;
    }

    if (listen_fd_ >= 0) {
        ::shutdown(listen_fd_, SHUT_RDWR);
        ::close(listen_fd_);
        listen_fd_ = -1;
    }

    data_cv_.notify_all();

    if (server_thread_.joinable()) {
        server_thread_.join();
    }

    std::lock_guard<std::mutex> lock(clients_mutex_);
    for (auto& worker : client_workers_) {
        if (worker.thread.joinable()) {
            worker.thread.join();
        }
    }
    client_workers_.clear();
}

void WebServer::updateFrame(const cv::Mat& bgr, const std::string& stats_json) {
    if (!running_.load()) return;
    if (bgr.empty()) return;

    std::vector<uchar> jpeg;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 80};
    if (!cv::imencode(".jpg", bgr, jpeg, params)) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        latest_jpeg_ = std::move(jpeg);
        stats_json_ = stats_json;
        ++frame_id_;
    }
    data_cv_.notify_all();
}

void WebServer::acceptLoop() {
    while (running_.load()) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
            if (!running_.load()) {
                break;
            }
            continue;
        }

        auto done = std::make_shared<std::atomic<bool>>(false);
        std::thread worker(&WebServer::handleClient, this, client_fd, done);
        {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            client_workers_.push_back({std::move(worker), done});
            for (auto it = client_workers_.begin(); it != client_workers_.end();) {
                if (it->done && it->done->load()) {
                    if (it->thread.joinable()) {
                        it->thread.join();
                    }
                    it = client_workers_.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
}

void WebServer::handleClient(int client_fd, const std::shared_ptr<std::atomic<bool>>& done) {
    struct DoneGuard {
        std::shared_ptr<std::atomic<bool>> done;
        ~DoneGuard() {
            if (done) {
                done->store(true);
            }
        }
    } done_guard{done};

    char buffer[1024];
    int n = ::recv(client_fd, buffer, sizeof(buffer) - 1, 0);
    if (n <= 0) {
        ::close(client_fd);
        return;
    }
    buffer[n] = '\0';

    std::istringstream req(buffer);
    std::string method;
    std::string path;
    req >> method >> path;

    if (method != "GET") {
        ::close(client_fd);
        return;
    }

    if (path == "/stream") {
        serveStream(client_fd);
    } else if (path == "/stats") {
        serveStats(client_fd);
    } else {
        serveIndex(client_fd);
    }

    ::close(client_fd);
}

void WebServer::serveIndex(int client_fd) {
    const char* body =
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>Cat Feeder</title>"
        "<style>body{font-family:Arial,sans-serif;margin:20px;background:#111;color:#eee}"
        "#stats{margin:10px 0;font-size:14px}img{max-width:100%;height:auto;border:1px solid #333}</style>"
        "</head><body>"
        "<h2>Cat Feeder Stream</h2>"
        "<div id=\"stats\">loading...</div>"
        "<img id=\"stream\" src=\"/stream\" />"
        "<script>"
        "async function tick(){"
        "const r=await fetch('/stats');"
        "if(r.ok){document.getElementById('stats').textContent=await r.text();}"
        "setTimeout(tick,500);}"
        "tick();"
        "</script></body></html>";

    std::ostringstream header;
    header << "HTTP/1.1 200 OK\r\n"
           << "Content-Type: text/html\r\n"
           << "Content-Length: " << std::strlen(body) << "\r\n"
           << "Connection: close\r\n\r\n";

    sendAll(client_fd, header.str().c_str(), header.str().size());
    sendAll(client_fd, body, std::strlen(body));
}

void WebServer::serveStats(int client_fd) {
    std::string stats;
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        stats = stats_json_;
    }
    if (stats.empty()) {
        stats = "no stats yet";
    }

    std::ostringstream header;
    header << "HTTP/1.1 200 OK\r\n"
           << "Content-Type: text/plain\r\n"
           << "Content-Length: " << stats.size() << "\r\n"
           << "Connection: close\r\n\r\n";

    sendAll(client_fd, header.str().c_str(), header.str().size());
    sendAll(client_fd, stats.c_str(), stats.size());
}

void WebServer::serveStream(int client_fd) {
    std::ostringstream header;
    header << "HTTP/1.1 200 OK\r\n"
           << "Cache-Control: no-cache\r\n"
           << "Pragma: no-cache\r\n"
           << "Connection: close\r\n"
           << "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
    if (!sendAll(client_fd, header.str().c_str(), header.str().size())) {
        return;
    }

    std::uint64_t last_frame_id = 0;
    while (running_.load()) {
        std::vector<uchar> jpeg;
        std::uint64_t frame_id = 0;
        {
            std::unique_lock<std::mutex> lock(data_mutex_);
            data_cv_.wait_for(lock, std::chrono::milliseconds(200), [&] {
                return frame_id_ != last_frame_id || !running_.load();
            });
            if (!running_.load()) break;
            jpeg = latest_jpeg_;
            frame_id = frame_id_;
        }

        if (jpeg.empty()) {
            continue;
        }
        if (frame_id == last_frame_id) {
            continue;
        }
        last_frame_id = frame_id;

        std::ostringstream part;
        part << "--frame\r\n"
             << "Content-Type: image/jpeg\r\n"
             << "Content-Length: " << jpeg.size() << "\r\n\r\n";

        if (!sendAll(client_fd, part.str().c_str(), part.str().size())) {
            break;
        }
        if (!sendAll(client_fd, jpeg.data(), jpeg.size())) {
            break;
        }
        if (!sendAll(client_fd, "\r\n", 2)) {
            break;
        }
    }
}

bool WebServer::sendAll(int fd, const void* data, std::size_t size) {
    const char* ptr = static_cast<const char*>(data);
    std::size_t sent = 0;
    while (sent < size) {
        ssize_t n = ::send(fd, ptr + sent, size - sent, 0);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return false;
        }
        sent += static_cast<std::size_t>(n);
    }
    return true;
}

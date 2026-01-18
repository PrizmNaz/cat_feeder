#pragma once
#include <opencv2/opencv.hpp>
#include <string>

class Camera {
    public: 
        explicit Camera(int index = 0);

        bool open();
        bool read(cv::Mat& outFrame);

        void request(int width, int height, double fps);
        std::string info() const;

    private: 
        int index_ = 0;
        int req_w_ = 640;
        int req_h_ = 480;
        double req_fps_ = 30.0;

        cv::VideoCapture cap_;
};
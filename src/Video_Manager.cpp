#include "Video_Manager.h"

VideoManager::VideoManager(std::string source_type, std::string video_source): mFrameSize(cv::Size(256,256)){
    if(source_type == "webcam"){
        this->mSource = new cv::VideoCapture(0);
    }
    else{
        this->mSource = new cv::VideoCapture(video_source);
    }
    assert(this->mSource->isOpened());
}

void VideoManager::next_frame() {
    bool frame_read_success = this->mSource->read(this->mFrame);
    if(!frame_read_success){
        std::cout<<"Failed to read from video source\n";
        exit(0);
    }
    if(this->mCropRect && this->mCropRect->area()!=0) {
        this->mProcessedFrame = this->mFrame(*this->mCropRect);
        cv::resize(this->mProcessedFrame, this->mProcessedFrame, this->mFrameSize, 0, 0, cv::INTER_CUBIC);
    }
}

cv::Mat* VideoManager::get_frame() {
    return &(this->mFrame);
}

cv::Mat* VideoManager::get_processed_frame() {
    return &(this->mProcessedFrame);
}

void VideoManager::crop_source(int top_left_x, int top_left_y, int bottom_right_x, int bottom_right_y) {
    if(top_left_x == -1){
        if(bottom_right_x == -1){
            delete this->mCropRect;
            this->mCropRect = nullptr;
            return;
        }
        assert(top_left_y == -1);
        top_left_x = this->mCropRect->x;
        top_left_y = this->mCropRect->y;
    }
    if(this->mCropRect == nullptr){
        this->mCropRect = new cv::Rect(top_left_x, top_left_y, bottom_right_x-top_left_x, bottom_right_y-top_left_y);
    }
    else{
        *this->mCropRect = cv::Rect(top_left_x, top_left_y, bottom_right_x-top_left_x, bottom_right_y-top_left_y);
    }
}

void VideoManager::draw_crop_border() {
    if(this->mCropRect){
        cv::rectangle(this->mFrame, *this->mCropRect, cv::Scalar(0,0,255), 2);
    }
}

VideoManager::~VideoManager() {
    if(this->mCropRect) delete this->mCropRect;
    //delete this->mSource;
}
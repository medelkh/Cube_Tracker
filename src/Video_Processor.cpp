#include "Video_Processor.h"

VideoProcessor::VideoProcessor(VideoManager *source) : mSource(source), mSobelMask((cv::Mat_<float>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1)){

}

void VideoProcessor::retrieve_frame() {
    this->mFrame = this->mSource->get_processed_frame();
}

void VideoProcessor::process_frame() {
    this->retrieve_frame();
    this->grayscale_frame();
}

void VideoProcessor::grayscale_frame() {
    cv::cvtColor(*this->mFrame, this->mGrayFrame, cv::COLOR_BGR2GRAY);
}

void VideoProcessor::edge_frame(){
    this->mGrayFrame.convertTo(this->mEdgeFrame, CV_32F, 1.0 / 255.0);

    //apply the sobel mask horizontally and vertically
    cv::Mat dx, dy;
    cv::filter2D(this->mEdgeFrame, dx, -1, this->mSobelMask);
    cv::filter2D(this->mEdgeFrame, dy, -1, this->mSobelMask.t());

    // Calculate magnitude and orientation
    cv::cartToPolar(dx, dy, this->mEdgeFrame, cv::noArray(), true);

    // Normalize the magnitude to [0, 1]
    cv::normalize(this->mEdgeFrame, this->mEdgeFrame, 0, 1, cv::NORM_MINMAX);
}

VideoProcessor::~VideoProcessor() {

}
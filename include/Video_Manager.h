#ifndef VideoManager_H
#define VideoManager_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>


class VideoManager {
private:
    cv::VideoCapture* mSource; //the video source (either webcam or a video file)

    cv::Mat mFrame; //the current unprocessed frame of the source
    cv::Mat mProcessedFrame; //the current frame after being cropped then resized

    cv::Rect* mCropRect{nullptr}; //the rectangle to which the frame will be cropped automatically into mCroppedFrame after each read
    cv::Size mFrameSize; //the size to which the frame will be resized into mResizedFrame after each read

public:
    //Constructor for the Video Manager, takes as a parameter the source_type, which could be either
    //"webcam" or "video", as well as an optional parameter video_source in case the source type is a video
    VideoManager(std::string source_type = "webcam", std::string video_source = "");

    //Moves to the next frame in the video source
    void next_frame();

    //Returns a pointer to the current unfiltered frame of the video source
    cv::Mat* get_frame();

    //Returns a pointer to the current frame after being cropped then resized
    cv::Mat* get_processed_frame();

    //Crops the video source, if all parameters are -1 then it will reset the crop rect to null, otherwise if
    //top_left_x = -1, then it will consider the already existing top left point of mCropRect
    void crop_source(int top_left_x, int top_left_y, int bottom_right_x, int bottom_right_y);

    //Draws a rectangle around the cropped area
    void draw_crop_border();

    //Video Manager Destructor
    ~VideoManager();
};

#endif
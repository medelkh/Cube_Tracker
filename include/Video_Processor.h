#ifndef VideoProcessor_H
#define VideoProcessor_H

#include <opencv2/opencv.hpp>
#include "Video_Manager.h"

class VideoProcessor {
private:
    VideoManager* mSource;

    cv::Mat mSobelMask; //the 3x3 sobel mask

    cv::Mat* mFrame;
    cv::Mat mGrayFrame;
    cv::Mat mEdgeFrame; //sobel transform for edge detection

public:
    //Constructor
    VideoProcessor(VideoManager* source);

    //gets the frame from the source then performs all the calculations on it
    void process_frame();

    //gets the frame from the source
    void retrieve_frame();

    //gets the gray scale frame from the source frame
    void grayscale_frame();

    //performs the sobel transformation on the grayscale frame
    void edge_frame();

    //Destructor
    ~VideoProcessor();
};


#endif

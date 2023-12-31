#ifndef VideoProcessor_H
#define VideoProcessor_H

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include "Video_Manager.h"
#include "utils.h"

class VideoProcessor {
private:
    VideoManager* mSource;

    torch::jit::script::Module mBbModel;
    torch::jit::script::Module mUcModel;

    cv::Size mCubeFrameSize;

    cv::Rect mCubeRect; //the detected cube Rect
    std::pair<Quad, float> mUpperCenter; //the predicted upper center as well as the confidence level

    cv::Mat mSobelMask; //the 3x3 sobel mask

    cv::Mat* mFrame; //original frame
    cv::Mat mGrayFrame; //grayscale frame
    cv::Mat mEdgeFrame; //sobel transform for edge detection
    cv::Mat mCubeEdgeFrame; //the detected cube frame with the edge filter applied on it
    cv::Mat mCubeFrame; //the detected cube frame cropped from the unfiltered original frame

    torch::Device mDevice; //CUDA (GPU) or CPU

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

    //gets the cube's bounding rect from the Bounding Box model and stores it in mCubeRect
    void calc_cube_rect();

    //stores the cropped cube's bounding box in both mCubeEdgeFrame and mCubeFrame
    void set_cube_frames();

    //gets the upper center's corners' position with a confidence level from the Upper Center model and stores it in mUpperCenter
    void calc_upper_center();

    //draws the bounding box of the cube on the originally retrieved frame
    void draw_cube_bb();

    //draws the predicted upper center's position
    void draw_upper_center();

    //returns a pointer to the processed frame
    cv::Mat* get_frame();

    //loads the models' parameters from the .pth files
    void load_models();

    //Destructor
    ~VideoProcessor();
};


#endif

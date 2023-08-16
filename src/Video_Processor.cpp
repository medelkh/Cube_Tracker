#include "Video_Processor.h"

VideoProcessor::VideoProcessor(VideoManager *source) : mSource(source), mCubeFrameSize(cv::Size(128,128)), mUpperCenter({Quad(), 0.}),
                                                       mSobelMask((cv::Mat_<float>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1)),
                                                       mDevice(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU){
    if(torch::cuda::is_available()){
        std::cout << "Working with CUDA" << std::endl;
    }
    else{
        std::cout << "CUDA unavailable. CPU will be used instead." << std::endl;
    }

    this->load_models();

    this->mBbModel.to(this->mDevice);
    this->mUcModel.to(this->mDevice);
}

void VideoProcessor::retrieve_frame() {
    this->mFrame = this->mSource->get_processed_frame();
}

void VideoProcessor::process_frame() {
    this->retrieve_frame();
    this->grayscale_frame();
    this->edge_frame();
    this->calc_cube_rect();
    this->set_cube_frames();
    this->calc_upper_center();
}

void VideoProcessor::grayscale_frame() {
    cv::cvtColor(*this->mFrame, this->mGrayFrame, cv::COLOR_BGR2GRAY);
}

void VideoProcessor::edge_frame(){
    this->mGrayFrame.convertTo(this->mEdgeFrame, CV_32F, 1./255.);

    //apply the sobel mask horizontally and vertically
    cv::Mat dx, dy;
    cv::filter2D(this->mEdgeFrame, dx, -1, this->mSobelMask);
    cv::filter2D(this->mEdgeFrame, dy, -1, this->mSobelMask.t());

    // Calculate magnitude and orientation
    cv::magnitude(dx, dy, this->mEdgeFrame);

    cv::normalize(this->mEdgeFrame, this->mEdgeFrame, 0, 1, cv::NORM_MINMAX);
}

void VideoProcessor::calc_cube_rect() {
    //converting the frame from cv::Mat to a torch::jit::IValue
    torch::jit::IValue edge_frame_tensor = torch::jit::IValue(torch::from_blob(this->mEdgeFrame.data, {1, 256, 256}, torch::kFloat32).to(this->mDevice));
    //applying the Bounding Box model to obtain the bounding box tensor
    at::Tensor bounding_box_tensor = this->mBbModel.forward({edge_frame_tensor}).toTensor();
    //converting the bounding box tensor to an array while
    //computing the absolute bounding box pixel coordinates (integers in [0, 256[) from the relative coordinates (floats in [0, 1])
    int bounding_box[4];
    for(int i=0; i < 4; i++) bounding_box[i] = int(255. * bounding_box_tensor[i].item<float>());
    //upscaling the bounding box by a little to account for the model's imprecision
    bounding_box[0] = std::max(bounding_box[0]-8, 0);
    bounding_box[1] = std::max(bounding_box[1]-8, 0);
    bounding_box[2] = std::min(bounding_box[2]+8, 255);
    bounding_box[3] = std::min(bounding_box[3]+8, 255);
    //storing the final cube rect in mCubeRect
    this->mCubeRect = cv::Rect(bounding_box[0], bounding_box[1], bounding_box[2]-bounding_box[0], bounding_box[3]-bounding_box[1]);
}

void VideoProcessor::set_cube_frames() {
    //storing the unfiltered cube frame in mCubeFrame
    this->mCubeFrame = (*this->mFrame)(this->mCubeRect);
    cv::resize(this->mCubeFrame, this->mCubeFrame, this->mCubeFrameSize, 0, 0, cv::INTER_CUBIC);
    //storing the cube frame with the edge filter applied on it in mCubeEdgeFrame
    this->mCubeEdgeFrame = this->mEdgeFrame(this->mCubeRect);
    cv::resize(this->mCubeEdgeFrame, this->mCubeEdgeFrame, this->mCubeFrameSize, 0, 0, cv::INTER_CUBIC);
}

void VideoProcessor::calc_upper_center() {
    //converting the edge filtered cube frame from cv::Mat to a torch::jit::IValue
    torch::jit::IValue cube_edge_frame_tensor = torch::jit::IValue(torch::from_blob(this->mCubeEdgeFrame.data, {1, 128, 128}, torch::kFloat32).to(this->mDevice));
    //applying the Bounding Box model to obtain the bounding box tensor
    at::Tensor upper_center_tensor = this->mUcModel.forward({cube_edge_frame_tensor}).toTensor();
    //storing the upper center in mUpperCenter
    this->mUpperCenter = {Quad(cv::Point(mCubeRect.x + int(mCubeRect.width * upper_center_tensor[0].item<float>()),mCubeRect.y + int(mCubeRect.height * upper_center_tensor[1].item<float>())),
                               cv::Point(mCubeRect.x + int(mCubeRect.width * upper_center_tensor[2].item<float>()),mCubeRect.y + int(mCubeRect.height * upper_center_tensor[3].item<float>())),
                               cv::Point(mCubeRect.x + int(mCubeRect.width * upper_center_tensor[4].item<float>()),mCubeRect.y + int(mCubeRect.height * upper_center_tensor[5].item<float>())),
                               cv::Point(mCubeRect.x + int(mCubeRect.width* upper_center_tensor[6].item<float>()),mCubeRect.y + int(mCubeRect.height * upper_center_tensor[7].item<float>()))),
                               upper_center_tensor[8].item<float>()};
}

void VideoProcessor::draw_cube_bb() {
    cv::rectangle(*this->mFrame, this->mCubeRect, {0,0,255}, 2);
}

void VideoProcessor::draw_upper_center() {
    std::cout << "Upper Center condidence level : " << this->mUpperCenter.second << std::endl;
    for(int i=0; i<4; i++){
        cv::line(*mFrame, mUpperCenter.first[i], mUpperCenter.first[i+1], {0, 255, 0}, 2);
    }
}

cv::Mat* VideoProcessor::get_frame() {
    return this->mFrame;
}

void VideoProcessor::load_models() {
    this->mBbModel = torch::jit::load("./models/bb_model.pt");
    this->mUcModel = torch::jit::load("./models/uc_model.pt");
}

VideoProcessor::~VideoProcessor() {
}
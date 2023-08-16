#include "App.h"

App::App(std::string type, std::string file_path) {
    this->mSource = new VideoManager(type, file_path);
    this->mProcessor = new VideoProcessor(this->mSource);
}

void App::init() {
    cv::namedWindow(this->mWindowName);
    cv::setMouseCallback(this->mWindowName, this->EventHandle, (void*) this);
}

bool App::run_selection() {
    this->mSource->next_frame();
    //selecting the cropping area by hand
    while(true){
        this->mSource->next_frame(); //get the next frame from the video source
        this->mSource->draw_crop_border(); //draw the selected rect so far

        cv::imshow(this->mWindowName, *(this->mSource->get_frame())); //display the unprocessed frame with the selected rect drawn on it

        int key = cv::waitKey(1000./(this->mFPS));
        if(key == (int)'s' && this->mPhase==SELECTION_DONE) break; //move to the cube tracking phase once the cropping rect is selected and "s" is pressed
        else if(key == (int)'q') return false; //if q is pressed, quit
    }
    return true;
}

bool App::run_tracking() {
    this->mPhase = CUBE_TRACKING;
    //displaying the cropped frame
    while(true){
        this->mSource->next_frame();

        this->mProcessor->process_frame();
        this->mProcessor->draw_cube_bb();
        this->mProcessor->draw_upper_center();

        cv::imshow(this->mWindowName, *(this->mProcessor->get_frame()));

        int key = cv::waitKey(1000./(this->mFPS));
        if(key == (int)'q') return false;
    }
    return true;
}


void App::run() {
    //run the selection phase of the app
    bool selection_success = this->run_selection();
    if(!selection_success) return;

    //run the cube tracking phase of the app
    bool tracking_success = this->run_tracking();
    if(!tracking_success) return;
}

void App::EventHandle(int event, int x, int y, int flags, void *app_object) {
    App* this_app = (App*) app_object;
    if(this_app->mPhase == CUBE_TRACKING) return;
    if(event == cv::EVENT_LBUTTONDOWN){
        this_app->mSource->crop_source(x, y, x, y);
        this_app->mPhase = PENDING_SELECTION;
    }
    else if(event == cv::EVENT_RBUTTONDOWN){
        this_app->mSource->crop_source(-1, -1, -1, -1);
        this_app->mPhase = START;
    }
    else if(event == cv::EVENT_MOUSEMOVE && this_app->mPhase == PENDING_SELECTION){
        this_app->mSource->crop_source(-1, -1, x, y);
    }
    else if(event == cv::EVENT_LBUTTONUP){
        this_app->mPhase = SELECTION_DONE;
    }
}

void App::quit(){
    cv::destroyWindow(this->mWindowName);
}

App::~App(){
    delete this->mSource;
    delete this->mProcessor;
}
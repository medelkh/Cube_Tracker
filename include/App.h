#ifndef App_H
#define App_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "Video_Manager.h"
#include "Video_Processor.h"
#include "utils.h"

enum Phase{
    START,
    PENDING_SELECTION,
    SELECTION_DONE,
    CUBE_TRACKING
};

class App
{
private:
	VideoManager* mSource;
    VideoProcessor* mProcessor;
    std::string mWindowName{"Cube Tracker"};
    Phase mPhase{START};
    double mFPS{60};

public:
	//App constructor
    App(std::string type, std::string file_path = "");

	//Initialize all the needed models and doing the rest of the necessary setup
	void init();

	//Run the application
	void run();

    //Quit the application
    void quit();

    //Handles the mouse events
    static void EventHandle(int event, int x, int y, int flags, void* app_object);

    //Runs the crop area selection phase of the app
    bool run_selection();

    //Runs the cube tracking phase of the app
    bool run_tracking();

    //App destructor
    ~App();
};

#endif
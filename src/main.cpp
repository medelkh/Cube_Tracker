#include <iostream>
#include "App.h"

int main(){
    App* app;
    int type;
    std::cout << "The source type is:\n  1/webcam\n  2/video" << std::endl;
    std::cin >> type;
    if(type == 1) {
        app = new App("webcam");
    }
    else{
        std::string video_path;
        std::cout << "Enter the video path: " << std::endl;
        std::cin >> video_path;
        app = new App("video", video_path);
    }
    app->init();
    app->run();
    app->quit();
    delete app;

    return 0;
}
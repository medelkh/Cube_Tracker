#include "App.h"

int main(){
    App* app = new App();
    app->init();
    app->run();
    app->quit();
    delete app;

    return 0;
}
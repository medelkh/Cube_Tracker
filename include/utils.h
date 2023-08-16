#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>

//Quadrilateral struct
struct Quad{
    cv::Point points[4]; //the points of the quadrilateral: top left, top right, bottom right, bottom left

    Quad(cv::Point top_left = cv::Point(), cv::Point top_right = cv::Point(), cv::Point bottom_left = cv::Point(), cv::Point bottom_right = cv::Point()){
        points[0] = top_left;
        points[1] = top_right;
        points[2] = bottom_left;
        points[3] = bottom_right;
    }

    cv::Point& operator[] (int i){
        return points[i%4];
    }

    cv::Point& top_left(){
        return points[0];
    }

    cv::Point& top_right(){
        return points[1];
    }

    cv::Point& bottom_right(){
        return points[2];
    }

    cv::Point& bottom_left(){
        return points[3];
    }
};


#endif

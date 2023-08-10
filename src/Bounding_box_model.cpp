#include "Bounding_box_model.h"

BoundingBoxModelImpl::BoundingBoxModelImpl(std::string params_path) : torch::nn::Module(),
                                                        parameters_path(params_path),
                                                      conv1(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 3))),
                                                      maxpool1(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))),
                                                      conv2(torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 24, 3))),
                                                      maxpool2(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))),
                                                      conv3(torch::nn::Conv2d(torch::nn::Conv2dOptions(24, 32, 5))),
                                                      maxpool3(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(4).stride(4))),
                                                      fc(torch::nn::Linear(torch::nn::LinearOptions(32*14*14, 4)))
{
}

void BoundingBoxModelImpl::init(){

    //defining the model's architecture
    //first conv layer
    this->register_module("bbconv1", this->conv1);
    this->register_module("bbmaxpool1", this->maxpool1);
    //second conv layer
    this->register_module("bbconv2", this->conv2);
    this->register_module("bbmaxpool2", this->maxpool2);
    //third conv layer
    this->register_module("bbconv3", this->conv3);
    this->register_module("bbmaxpool3", this->maxpool3);
    //FC layer
    this->register_module("bbfc", this->fc);
}

torch::Tensor BoundingBoxModelImpl::forward(torch::Tensor x) {
    //Applying the first conv layer
    x = this->maxpool1(torch::relu(this->conv1(x)));

    //pass through second conv layer
    x = this->maxpool2(torch::relu(this->conv2(x)));

    //pass through third conv layer
    x = this->maxpool3(torch::relu(this->conv3(x)));

    //flatten the 2d filters into a 1d tensor
    torch::flatten(x);

    //pass through FC layer
    x = torch::sigmoid(this->fc(x));

    return x;
}

std::string BoundingBoxModelImpl::get_parameters_path() {
    return this->parameters_path;
}


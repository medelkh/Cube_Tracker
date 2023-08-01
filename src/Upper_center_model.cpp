#include "Upper_center_model.h"

UpperCenterModelImpl::UpperCenterModelImpl(std::string params_path) : parameters_path(params_path),
                                                              conv1(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, 5))),
                                                              activ1(torch::nn::ReLU()),
                                                              maxpool1(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(3))),
                                                              conv2(torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 15, 4))),
                                                              activ2(torch::nn::ReLU()),
                                                              maxpool2(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))),
                                                              fc1(torch::nn::Linear(torch::nn::LinearOptions(15*19*19, 16))),
                                                              activFC1(torch::nn::ReLU()),
                                                              fc2(torch::nn::Linear(torch::nn::LinearOptions(16, 9))),
                                                              activFC2(torch::nn::Sigmoid())
{
    //defining the model's architecture
    //first conv layer
    register_module("conv1", this->conv1);
    register_module("activ1", this->activ1);
    register_module("maxpool1", this->maxpool1);
    //second conv layer
    register_module("conv2", this->conv2);
    register_module("activ2", this->activ2);
    register_module("maxpool2", this->maxpool2);
    //first FC layer
    register_module("fc1", this->fc1);
    register_module("activeFC1", this->activFC1);
    //second FC layer
    register_module("fc2", this->fc2);
    register_module("activeFC2", this->activFC2);
}

torch::Tensor UpperCenterModelImpl::forward(torch::Tensor x) {
    //pass through first conv layer
    x = this->conv1(x);
    x = this->activ1(x);
    x = this->maxpool1(x);

    //pass through second conv layer
    x = this->conv2(x);
    x = this->activ2(x);
    x = this->maxpool2(x);

    //flatten the 2d filters into a 1d tensor
    torch::flatten(x);

    //pass through the first FC layer
    x = this->fc1(x);
    x = this->activFC1(x);

    //pass through the second FC layer
    x = this->fc2(x);
    x = this->activFC2(x);

    return x;
}


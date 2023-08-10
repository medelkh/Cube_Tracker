#include "Upper_center_model.h"

UpperCenterModelImpl::UpperCenterModelImpl(std::string params_path) : parameters_path(params_path),
                                                              conv1(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, 5))),
                                                              maxpool1(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(3))),
                                                              conv2(torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 15, 4))),
                                                              maxpool2(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))),
                                                              fc1(torch::nn::Linear(torch::nn::LinearOptions(15*19*19, 16))),
                                                              fc2(torch::nn::Linear(torch::nn::LinearOptions(16, 9)))
{
    //defining the model's architecture
    //first conv layer
    this->register_module("ucconv1", this->conv1);
    this->register_module("ucmaxpool1", this->maxpool1);
    //second conv layer
    this->register_module("ucconv2", this->conv2);
    this->register_module("ucmaxpool2", this->maxpool2);
    //first FC layer
    this->register_module("ucfc1", this->fc1);
    //second FC layer
    this->register_module("ucfc2", this->fc2);
}

torch::Tensor UpperCenterModelImpl::forward(torch::Tensor x) {
    //Applying the first conv layer
    x = this->maxpool1(torch::relu(this->conv1(x)));

    //Applying the second conv layer
    x = this->maxpool2(torch::relu(this->conv2(x)));

    //Flatten the 2d filters into a 1d tensor
    torch::flatten(x);

    //Applying the first FC layer
    x = torch::relu(this->fc1(x));

    //Applying the second FC layer
    x = torch::sigmoid(this->fc2(x));

    return x;
}

std::string UpperCenterModelImpl::get_parameters_path() {
    return this->parameters_path;
}

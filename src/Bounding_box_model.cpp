#include "Bounding_box_model.h"

BoundingBoxModelImpl::BoundingBoxModelImpl(std::string params_path) : parameters_path(params_path),
                                                              conv1(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 3))),
                                                              activ1(torch::nn::ReLU()),
                                                              maxpool1(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))),
                                                              conv2(torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 24, 3))),
                                                              activ2(torch::nn::ReLU()),
                                                              maxpool2(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))),
                                                              conv3(torch::nn::Conv2d(torch::nn::Conv2dOptions(24, 32, 5))),
                                                              activ3(torch::nn::ReLU()),
                                                              maxpool3(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(4).stride(4))),
                                                              fc(torch::nn::Linear(torch::nn::LinearOptions(32*14*14, 4))),
                                                              activFC(torch::nn::Sigmoid())
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
    //third conv layer
    register_module("conv3", this->conv3);
    register_module("activ3", this->activ3);
    register_module("maxpool3", this->maxpool3);
    //FC layer
    register_module("fc", this->fc);
    register_module("activeFC", this->activFC);
}

torch::Tensor BoundingBoxModelImpl::forward(torch::Tensor x) {
    //pass through first conv layer
    x = this->conv1(x);
    x = this->activ1(x);
    x = this->maxpool1(x);

    //pass through second conv layer
    x = this->conv2(x);
    x = this->activ2(x);
    x = this->maxpool2(x);

    //pass through third conv layer
    x = this->conv3(x);
    x = this->activ3(x);
    x = this->maxpool3(x);

    //flatten the 2d filters into a 1d tensor
    torch::flatten(x);

    //pass through FC layer
    x = this->fc(x);
    x = this->activFC(x);

    return x;
}


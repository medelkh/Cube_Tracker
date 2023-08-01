#ifndef UpperCenterModel_H
#define UpperCenterModel_H

#include <torch/torch.h>


class UpperCenterModelImpl : public torch::nn::Module{
private:
    std::string parameters_path;
    //first convolutional layer
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::ReLU activ1{nullptr};
    torch::nn::MaxPool2d maxpool1{nullptr};
    //second convolutional layer
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::ReLU activ2{nullptr};
    torch::nn::MaxPool2d maxpool2{nullptr};
    //first FC layer
    torch::nn::Linear fc1{nullptr};
    torch::nn::ReLU activFC1{nullptr};
    //second FC layer
    torch::nn::Linear fc2{nullptr};
    torch::nn::Sigmoid activFC2{nullptr};

    //model's parameters

public:
    UpperCenterModelImpl(std::string params_path);
    torch::Tensor forward(torch::Tensor x);

    //Initialize the model
    void init();
};

TORCH_MODULE(UpperCenterModel);

#endif
#ifndef UpperCenterModel_H
#define UpperCenterModel_H

#include <torch/torch.h>


class UpperCenterModelImpl : public torch::nn::Module{
private:
    std::string parameters_path;
    //First convolutional layer
        torch::nn::Conv2d conv1{nullptr};
        //ReLU activation
        torch::nn::MaxPool2d maxpool1{nullptr};
    //Second convolutional layer
        torch::nn::Conv2d conv2{nullptr};
        //ReLU activation
        torch::nn::MaxPool2d maxpool2{nullptr};
    //First FC layer
        torch::nn::Linear fc1{nullptr};
        //ReLU activation
    //Second FC layer
        torch::nn::Linear fc2{nullptr};
        //Sigmoid activation

public:
    UpperCenterModelImpl(std::string params_path);
    torch::Tensor forward(torch::Tensor x);

    //getting the path to the model's parameters
    std::string get_parameters_path();
};

TORCH_MODULE(UpperCenterModel);

#endif
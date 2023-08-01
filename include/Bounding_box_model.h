#ifndef BoundingBoxModel_H
#define BoundingBoxModel_H

#include <torch/torch.h>


class BoundingBoxModelImpl : public torch::nn::Module{
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
    //third convolutional layer
    torch::nn::Conv2d conv3{nullptr};
    torch::nn::ReLU activ3{nullptr};
    torch::nn::MaxPool2d maxpool3{nullptr};
    //FC layer
    torch::nn::Linear fc{nullptr};
    torch::nn::Sigmoid activFC{nullptr};

    //model's parameters

public:
	BoundingBoxModelImpl(std::string params_path);
	torch::Tensor forward(torch::Tensor x);

	//Initialize the model
	void init();
};

TORCH_MODULE(BoundingBoxModel);

#endif

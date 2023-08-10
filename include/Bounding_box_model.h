#ifndef BoundingBoxModel_H
#define BoundingBoxModel_H

#include <torch/torch.h>


class BoundingBoxModelImpl : public torch::nn::Module{
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
    //Third convolutional layer
        torch::nn::Conv2d conv3{nullptr};
        //ReLU activation
        torch::nn::MaxPool2d maxpool3{nullptr};
    //FC layer
        torch::nn::Linear fc{nullptr};
        //Sigmoid activation

public:
	BoundingBoxModelImpl(std::string params_path);
    void init();
	torch::Tensor forward(torch::Tensor x);

	//getting the path to the model's parameters
	std::string get_parameters_path();
};

TORCH_MODULE(BoundingBoxModel);

#endif

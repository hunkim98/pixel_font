from collections import OrderedDict
import pickle
import numpy as np

from .layers import Affine, Relu, SoftmaxWithLoss
from .conv_layers import Convolution, Pooling
# The structure will be as follows:
# Input -> Conv -> ReLU -> Pooling -> Affine -> ReLU -> Affine -> Softmax -> Output
# You can have more than one convolutional layer, but for simplicity, we will only use one convolutional layer.
# ReLU is just a layer that acts like a neuron in the brain. It is a layer that applies the ReLU function to the input.
# If the incoming value is less than 0, then the output is 0. If the incoming value is greater than 0, then the output is the incoming value.
# This tells the neural network to only activate if the incoming value is greater than 0.

class SimpleConvNet:
    """
    Simple Convolutional Neural Network

    Structure:
    Input -> Conv --(W1)--> ReLU -> Pooling --(W2)--> Affine -> ReLU -> Affine --(W3)--> Softmax -> Output

    Parameters
    ----------
    input_size : 입력 크기（MNIST의 경우엔 784） - 28 * 28
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
    output_size : 출력 크기（MNIST의 경우엔 10）
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
        'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
        'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정

    You can have more than one convolutional layer, but for simplicity, we will only use one convolutional layer.

    """
    def __init__(self, input_dim=(1,28,28), \
                # if no filter_num is specified, 30 windows will be used for analyzing the image
                # 30 windows initialized with random weights will be used to analyze the image
                # each window will be a 5 by 5 window
                # each window will stride through the image with a stride of 1
                # each window will be padded with 0 pixels
                # each window will do its best to analyze the image to lessen the loss
                # Humans cannot know what each window is looking for, but we can know that each window is looking for something
                # The top sentence is the reason we call neural networks a black box!
                 conv_param = {'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1}, \
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        # input_dim is the dimension of the input data
        # conv_param is the parameters for the convolutional layer
        # hidden_size is the number of neurons in the hidden layer
        # output_size is the number of neurons in the output layer
        # weight_init_std is the standard deviation of the weights
        filter_num = conv_param['filter_num']
        # filter size is the size of the filter 
        # it is a square,so the actual pixel count is filter_size * filter_size
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        # the output size of the convolutional layer is calculated as follows:
        # (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        # the output size of the pooling layer is calculated as follows:
        # (input_size - pooling_size) / pooling_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))
        # the pooling layer is a layer that reduces the size of the image
        # the pooling layer is a layer that extracts the most important features from the image
        # the pooling layer is a layer that reduces the size of the image by extracting the most important features from the image

        # Initialize weights
        self.params = {}
        # W1 is the weights for the convolutional layer to Relu layer
        self.params["W1"] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        
        self.params["b1"] = np.zeros(filter_num)

        # W2 is the weights for pooling layer to Affine layer
        self.params["W2"] = weight_init_std * \
                            np.random.randn(pool_output_size, hidden_size)
        
        self.params["b2"] = np.zeros(hidden_size)

        # W3 is the weights for the Affine layer to the softmax(=output) layer
        self.params["W3"] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        
        self.params["b3"] = np.zeros(output_size)

        # Generate layers
        self.layers = OrderedDict()
        # Convolutional layer
        self.layers["Conv1"] = Convolution(self.params["W1"], self.params["b1"], \
                                            conv_param["stride"], conv_param["pad"])
        # ReLU layer
        self.layers["Relu1"] = Relu()
        # Pooling layer
        self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)
        # Affine layer
        self.layers["Affine1"] = Affine(self.params["W2"], self.params["b2"])
        # ReLU layer
        self.layers["Relu2"] = Relu()
        # Affine layer
        self.layers["Affine2"] = Affine(self.params["W3"], self.params["b3"])

        self.last_layer = SoftmaxWithLoss()
    
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    def accuracy(self, x, t, batch_size=100):
        # if the batch size is not specified, then the batch size is 100
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        # we divide the data into batches
        # we loop through each batch
        # we calculate the accuracy of each batch
        # we add the accuracy of each batch to the total accuracy
        # we divide the total accuracy by the number of batches to get the average accuracy
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            
            # we predict the output of the neural network
            y = self.predict(tx)
            # we predict the index of the maximum value in the output of the neural network
            y = np.argmax(y, axis=1)
            # we calculate the number of correct predictions
            acc += np.sum(y == tt) 
        
        # we divide the number of correct predictions by the number of data to get the accuracy
        return acc / x.shape[0]
    
    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        
        # backward
        dout = 1
        # we calculate the gradient of each layer
        # we
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        # we reverse the order of the layers
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        # we save the gradients in a dictionary
        grads = {}
        # W1 is the weights for the convolutional layer to Relu layer
        grads["W1"], grads["b1"] = self.layers["Conv1"].dW, self.layers["Conv1"].db
        # W2 is the weights for pooling layer to Affine layer
        grads["W2"], grads["b2"] = self.layers["Affine1"].dW, self.layers["Affine1"].db
        # W3 is the weights for the Affine layer to the softmax(=output) layer
        grads["W3"], grads["b3"] = self.layers["Affine2"].dW, self.layers["Affine2"].db
        
        return grads
    
    # this is for saving the parameters of the neural network
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    # this is for loading the parameters of the neural network
    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
                                           


                 
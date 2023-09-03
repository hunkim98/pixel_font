# Convolution layer is a layer for convolutional neural networks
# We use this layer to extract features from the image
# This layer is especially used when we need to preserve the spatial relationship of the original data
# For instance, if we are working with image data, we need to preserve the spatial relationship of the pixels
import numpy as np
from common.utils import col2im, im2col


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        # W is the filter weights
        # W is a matrix of size (FN, C, FH, FW)
        # FN is the number of filters
        # the filter number is an arbitrary number that we can choose
        # the more the filter number, the more features we can extract from the image
        # For instance if the input data is a 3 dimension image data, then the number of channels is 3
        # remember that in neural network we use W to represent weights
        # this W is what gets updated during backpropagation
        self.W = W
        # b is the bias
        self.b = b
        # stride is the unit in pixels that the filter moves
        self.stride = stride
        # pad is the number of pixels that we pad the image with
        self.pad = pad

        # 중간 데이터（backward 시 사용）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 가중치와 편향 매개변수의 기울기
        self.dW = None
        self.db = None
    
    def forward(self, x):
        # x is the input data
        # FN is the number of filters
        # C is the number of channels
        # FH is the filter height
        # FW is the filter width
        # C and C must be the same
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape

        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        # (FN, -1) means that we want to reshape the matrix into a matrix with FN rows and the number of columns is automatically calculated(thanks to -1 in the tuple)
        # We need to transpose the matrix because we want to calculate the dot product of the filter weights and the image data
        col_W = self.W.reshape(FN, -1).T
        # we are able to simply calculate the dot product of col and col_W because we have reshaped x to be easily calculated 
        # with our filter weights
        # using im2col, we were able to separate the image data into columns that are the same size as the filter (FH, FW)

        self.x = x
        self.col = col
        self.col_W = col_W     

        out = np.dot(col, col_W) + self.b

        # reshape the output to the desired shape
        # since we changed the shape of input data in i2col, we need to change it back to the original shape
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out
    
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

# Pooling is a layer that reduces the size of the image
# There are many modes for pooling: max pooling, average pooling, etc.
# In this example, we will use max pooling!
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        # pool_h is the pooling height
        # pool_w is the pooling width
        # stride is the unit in pixels that the filter moves
        # pad is the number of pixels that we pad the image with
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
    
    def forward(self, x):
        # N is the number of data
        # C is the number of channels
        # H is the height of the image
        # W is the width of the image
        N, C, H, W = x.shape

        # calculate the output size
        # H + 2 * pad is the size of the image after padding
        # we multiply pad by 2 because we are padding both sides of the image
        # filter is the size of the filter (Remember that the filter strids(=moves) along the image to calculate the output)
        # out_h is the height of the output after the filter has finished striding along the image
        # out_w is the width of the output after the filter has finished striding along the image
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # we use im2col to make the image into columns
        # we use reshape to make the columns into a 2 dimension array
        # we use argmax to find the index of the maximum value in each column
        # we use max to find the maximum value in each column (This is not the only way to do pooling!)
        # we use reshape to make the array into the desired shape
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        # we save the index of the maximum value in each column
        # we will use this index to calculate the gradient in the backward pass
        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
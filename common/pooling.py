import numpy as np
from common.utils import im2col

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
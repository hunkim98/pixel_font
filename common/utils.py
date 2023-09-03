import numpy as np

# This is a function that makes it easy for us to calculate CNN
# img 2 col changes the 4 dimension image data(= multiple 3 dimension[RGB] image data) into a 2 dimension data array customized for CNN filter
# we make the image to columns based on the filter size and stride

# input_data: 4 dimension data (N, C, H, W) N: number of data, C: number of channels, H: height, W: width
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape

    # calculate the output size
    # H + 2 * pad is the size of the image after padding
    # we multiply pad by 2 because we are padding both sides of the image
    # filter is the size of the filter (Remember that the filter strids(=moves) along the image to calculate the output)
    # out_h is the height of the output after the filter has finished striding along the image
    # out_w is the width of the output after the filter has finished striding along the image
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # since we are padding a 4 dimension data, we pad the 2nd and 3rd dimension
    # the first (0,0) and second (0,0) indicate that there is no padding for the first and second dimension
    # the first dimension has to do with the number of data, and the second dimension has to do with the number of channels
    # We do not need padding for these two dimensions since wour main focus is the image data
    # Remind yourself that the tuple is the format of (before, after) padding
    # if one of the data in the third dimension was [1, 2, 3, 4, 5], and we set the pad to 2, then the result would be [0, 0, 1, 2, 3, 4, 5, 0, 0]
    # 'constant' mode means that the padding is done with a constant value
    # constant_values=0 means that the padding is done with 0
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant', constant_values=0)

    # col is the array that we will store the vectorized(1 dimensional) image data 
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        # y_max is the maximum value of y
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            # y:ymax:stride means that we are selecting values from y to y_max every stride
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # The number in the tuple indicates the axis that we want to transpose
    # The first 0 indicates that we do not want to transpose the first dimension
    # 4, 5, 1, 2, 3 indicates that we want to reorder the reamining dimensions in the order of 4, 5, 1, 2, 3
    # The goal of the transpose is to amke the dimensions associated with the filter's height and width
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


# This is a function that makes the columns that we got from im2col into the original image data
# This is necessary because we need to calculate the gradient of the filter weights
# We use this in backpropagation
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
# This is the 1D version of the code from:
# Schubert, S., Neubert, P., PÃ¶schmann, J. & Protzel, P. (2019) Circular Convolutional Neural
# Networks for Panoramic Images and Laser Data. In Proc. of Intelligent Vehicles Symposium (IV)
# Link: https://www.tu-chemnitz.de/etit/proaut/en/research/rsrc/ccnn/code/ccnn_layers.py

from tensorflow.keras.layers import Conv1D, Cropping1D, Concatenate


def CConv1D(filters, kernel_size, strides=1, activation='linear', padding='valid',
            kernel_initializer='glorot_uniform', kernel_regularizer=None):

    def CConv1D_inner(x):
        in_width = int(x.get_shape()[1])

        if in_width % strides == 0:
            pad_along_width = max(kernel_size - strides, 0)
        else:
            pad_along_width = max(kernel_size - (in_width % strides), 0)

        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        # left and right side for padding
        pad_left = Cropping1D(cropping=(in_width-pad_left, 0))(x)
        pad_right = Cropping1D(cropping=(0, in_width-pad_right))(x)

        # add padding to incoming image
        conc = Concatenate(axis=1)([pad_left, x, pad_right])

        # perform the circular convolution
        cconv1d = Conv1D(filters=filters, kernel_size=kernel_size,
                         strides=strides, activation=activation,
                         padding='valid',
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer)(conc)

        # return circular convolution layer
        return cconv1d

    return CConv1D_inner

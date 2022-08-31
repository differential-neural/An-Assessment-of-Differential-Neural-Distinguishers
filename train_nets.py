import numpy as np
from pickle import dump
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2

from lib.CConv1D import CConv1D
from make_train_data import make_train_data

# +++ Based on train_nets.py by Gohr +++

bs = 5000
wdir = './freshly_trained_nets/'


def cyclic_lr(num_epochs, high_lr, low_lr):
    return lambda i: low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)


def make_checkpoint(file):
    return ModelCheckpoint(file, monitor='val_loss', save_best_only=True)


# make residual tower of convolutional blocks
def make_resnet(
        num_blocks=2, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=16, ks=3, depth=5, reg_param=0.0001,
        final_activation='sigmoid', cconv=False
):
    Conv = CConv1D if cconv else Conv1D  # Check if we use circular convolutions
    # Input and preprocessing layers
    inp = Input(shape=(num_blocks * word_size * 2,))
    rs = Reshape((2 * num_blocks, word_size))(inp)
    perm = Permute((2, 1))(rs)
    # add a single residual layer that will expand the data to num_filters channels
    # this is a bit-sliced layer
    conv0 = Conv(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation('relu')(conv0)
    # add residual blocks
    shortcut = conv0
    for i in range(depth):
        conv1 = Conv(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])
    # add prediction head
    flat1 = Flatten()(shortcut)
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(flat1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2)
    model = Model(inputs=inp, outputs=out)
    return model


def train_distinguisher(
        cipher, diff, n_train_samples=10**7, n_val_samples=10**6, n_epochs=80, depth=10, n_neurons=64, kernel_size=3,
        n_filters=32, reg_param=10 ** -5, lr_high=0.002, lr_low=0.0001, cconv=False, calc_back=0
):
    n_rounds = cipher.get_n_rounds()
    cipher_name = type(cipher).__name__
    result_base_name = f'{wdir}{cipher_name}_{n_rounds}_best_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    # create the network
    net = make_resnet(
        depth=depth, d1=n_neurons, d2=n_neurons, ks=kernel_size, num_filters=n_filters, reg_param=reg_param,
        cconv=cconv, word_size=cipher.get_word_size(), num_blocks=cipher.get_n_words()
    )
    net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    # generate training and validation data
    X, Y = make_train_data(n_train_samples, cipher, diff, calc_back)
    X_eval, Y_eval = make_train_data(n_val_samples, cipher, diff, calc_back)
    # set up model checkpoint
    check = make_checkpoint(f'{result_base_name}.h5')
    # create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10, lr_high, lr_low))
    # train and evaluate
    h = net.fit(X, Y, epochs=n_epochs, batch_size=bs, validation_data=(X_eval, Y_eval), callbacks=[lr, check])
    np.save(f'{result_base_name}_h.npy', h.history['val_acc'])
    np.save(f'{result_base_name}_h.npy', h.history['val_loss'])
    dump(h.history, open(f'{result_base_name}_hist.p', 'wb'))
    print(f'Best validation accuracy: {np.max(h.history["val_acc"])}, model saved as {result_base_name}.h5')
    return net, h

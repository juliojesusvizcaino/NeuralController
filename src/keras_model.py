#!/usr/bin/env python
import argparse
import glob

import h5py
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import RepeatVector, Dense, Dropout, Input, Convolution1D, LSTM, TimeDistributed
from keras.metrics import mean_squared_error, mean_absolute_error
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.data import StandardScaler


class MyModel(object):
    def __init__(self, x=None, y=None, x_val=None, y_val=None,
                 max_unroll=1502, name='model', save_dir='save/', log_dir='./logs'):
        self.x = x
        self.y = y
        self.x_val = x_val
        self.y_val = y_val
        self.model = self._keras_model(max_unroll)
        self.save_path = save_dir + name

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save = ModelCheckpoint(self.save_path + '-checkpoint.{epoch:06d}.hdf5')
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=10)
        self.callbacks = [save, tensorboard]

    def set_training_data(self, x, y):
        self.x = x
        self.y = y

    def set_validation_data(self, x, y):
        self.x_val = x
        self.y_val = y

    def _keras_model(self, max_unroll):
        inputs = Input(shape=(15,))

        x = RepeatVector(max_unroll)(inputs)
        x = TimeDistributed(Dense(64, init='normal'))(x)
        # x = Dropout(0.2)(x)
        x = LSTM(64, return_sequences=True, init='normal', dropout_U=0.2, dropout_W=0.2)(x)
        # x = Convolution1D(50, 3, border_mode='same', activation='softplus', init='normal')(x)
        # x = Dropout(0.1)(x)
        # x = Convolution1D(20, 3, border_mode='same', activation='softplus')(x)
        # x = Dropout(0.1)(x)
        x = TimeDistributed(Dense(50, activation='softplus', init='normal'))(x)
        x = TimeDistributed(Dense(50, activation='softplus', init='normal'))(x)
        # x = Dropout(0.1)(x)
        main_output = TimeDistributed(Dense(7, init='normal'), name='output')(x)
        mask_output = TimeDistributed(Dense(1, activation='sigmoid', init='normal'), name='mask')(x)

        model = Model(input=inputs, output=[main_output, mask_output])

        model.compile(optimizer='adam', loss='mae', sample_weight_mode='temporal',
                  metrics=[mean_absolute_error, mean_squared_error], loss_weights=[1., 1.])

        return model

    def fit(self, *args, **kwargs):
        self.model.fit(x=self.x, y=self.y, validation_data=[self.x_val, self.y_val],
                       callbacks=self.callbacks, *args, **kwargs)

    def load(self):
        file = max(glob.glob(self.save_path + '*'))
        self.model.load_weights(file)

    def resume(self, *args, **kwargs):
        self.load()
        self.fit(*args, **kwargs)


def parse():
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('-t', '--train', help='Begin training', action='store_true')
    parser.add_argument('-r', '--resume', help='Resume training', action='store_true')
    parser.add_argument('-f', '--filename', help='Name of the file')
    parser.add_argument('-n', '--nrollout', help='Number of rollouts', type=int, default=1502)
    parser.add_argument('-e', '--epoch', help='Number of epoch', type=int, default=500)
    return parser.parse_args()


def main():
    args = parse()
    n_rollout = args.nrollout
    n_epoch = args.epoch
    name = args.filename if args.filename is not None else 'model-' + str(n_rollout) + 'unroll'

    np.random.seed(1098)
    path = '../DataBase/data.hdf5'
    names = ['target_pos', 'target_speed', 'pos', 'vel', 'effort']
    with h5py.File(path, 'r') as f:
        (target_pos, target_speed, pos, vel, effort) = [[np.array(val) for val in f[name].values()] for name in names]

    x_target = np.array(target_pos)
    x_first = np.array([pos_[0] for pos_ in pos])
    x_speed = np.array(target_speed).reshape((-1, 1))
    aux_output = [np.ones(eff.shape[0]).reshape((-1, 1)) for eff in effort]

    x = np.concatenate((x_target, x_first, x_speed), axis=1)

    input_scaler = StandardScaler()
    x = input_scaler.fit_transform(x)
    output_scaler = StandardScaler()
    effort_concat = np.concatenate([a for a in effort], axis=0)
    output_scaler.fit(effort_concat)
    effort = [output_scaler.transform(eff) for eff in effort]

    y = pad_sequences(effort, padding='post', value=0.)
    aux_output = pad_sequences(aux_output, padding='post', value=0.)
    x, x_test, y, y_test, y_aux, y_aux_test = train_test_split(x, y, aux_output, test_size=0.2)

    model = MyModel(x, [y[:,:n_rollout,:], y_aux[:,:n_rollout,:]],
                    x_test, [y_test[:,:n_rollout,:], y_aux_test[:,:n_rollout,:]],
                    n_rollout, name=name)

    if not os.path.exists('save'):
        os.makedirs('save')

    if args.train:
        model.fit(nb_epoch=n_epoch, batch_size=32)
    elif args.resume:
        model.resume(nb_epoch=n_epoch, batch_size=32)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
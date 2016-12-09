#!/usr/bin/env python
import argparse
import os
import re
from glob import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers import RepeatVector, Dense, Input, TimeDistributed, Dropout, Convolution1D, GRU
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing.data import StandardScaler


class MyModel(object):
    def __init__(self, train, val, test=None, train_mask=None, val_mask=None, test_mask=None, max_unroll=None,
                 save_dir='save/', log_dir='logs/', img_dir='imgs/', *args, **kwargs):
        self.max_unroll = max_unroll if max_unroll is not None else train[1][0].shape[1]
        self.x, self.y = self._set_data(train)
        self.x_val, self.y_val = self._set_data(val)
        self.x_test, self.y_test = self._set_data(test) if test is not None else (None, None)
        self.train_mask = self._set_mask(train_mask)
        self.val_mask = self._set_mask(val_mask)
        self.test_mask = self._set_mask(test_mask)
        self.set_model(*args, **kwargs)
        self.save_path = save_dir if save_dir[-1] is '/' else save_dir + '/'
        self.img_path = img_dir if img_dir[-1] is '/' else img_dir + '/'

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)

        save = ModelCheckpoint(self.save_path + '-checkpoint.{epoch:06d}.hdf5')
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=10)
        earlyStopping = EarlyStopping(patience=5000)
        self.callbacks = [save, tensorboard, earlyStopping]

    def _set_data(self, data):
        x = data[0]
        y = [this_data[:, :self.max_unroll, :] for this_data in data[1]]
        return x, y

    def _set_mask(self, data):
        mask = [this_data[:, :self.max_unroll] for this_data in data]
        return mask

    def set_model(self, gru_width=100, gru_depth=2, dense_width=500, dense_depth=2, conv=False, conv_width=48,
                  conv_filter=3, dropout_fraction=0.5, *args, **kwargs):
        inputs = Input(shape=(15,))

        x = RepeatVector(self.max_unroll)(inputs)
        x = Dropout(dropout_fraction)(x)
        x = TimeDistributed(Dense(64, init='normal', activation='relu'), name='hidden_pre_GRU')(x)
        x = Dropout(dropout_fraction)(x)
        for i in range(gru_depth):
            x = GRU(gru_width, return_sequences=True, init='normal', activation='relu', dropout_U=dropout_fraction,
                    dropout_W=dropout_fraction, W_regularizer=l2(0.01), U_regularizer=l2(0.01))(x)
            x = Dropout(dropout_fraction)(x)
            if conv:
                x = Convolution1D(conv_width, conv_filter, border_mode='same')(x)
                x = Dropout(dropout_fraction)(x)

        for i in range(dense_depth):
            x = TimeDistributed(Dense(dense_width, activation='relu', init='normal', W_regularizer=l2(0.01)),
                                name='hidden_' + str(i))(x)
            x = Dropout(dropout_fraction)(x)

        torque_output = TimeDistributed(Dense(7, init='normal'), name='output_torque')(x)
        pos_output = TimeDistributed(Dense(7, init='normal'), name='output_pos')(x)
        vel_output = TimeDistributed(Dense(7, init='normal'), name='output_vel')(x)
        mask_output = TimeDistributed(Dense(1, activation='sigmoid', init='normal'), name='output_mask')(x)

        model = Model(input=inputs, output=[torque_output, pos_output, vel_output, mask_output])
        optimizer = Adam(decay=1e-6)
        model.compile(loss=['mae', 'mae', 'mae', 'binary_crossentropy'], sample_weight_mode='temporal',
                      loss_weights=[1., 0.1, 0.1, 1.], optimizer=optimizer, *args, **kwargs)

        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(x=self.x, y=self.y, validation_data=[self.x_val, self.y_val, self.val_mask],
                       sample_weight=self.train_mask, callbacks=self.callbacks, *args, **kwargs)

    def load(self):
        this_file = max(glob(self.save_path + '*'))
        self.model.load_weights(this_file)
        n = re.search(r'\D*(\d+)\.hdf5', this_file).group(1)
        return int(n)

    def resume(self, *args, **kwargs):
        init_epoch = self.load()
        self.fit(initial_epoch=init_epoch, *args, **kwargs)

    def save_figs(self, n=100):
        plot_names = ['train', 'cv', 'test']
        joint_names = ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2', 'mask']
        f, axs = plt.subplots(8, 1, figsize=(15, 20))
        for inp, outp, outp_aux, plot_name in zip((self.x, self.x_val, self.x_test),
                                                  (self.y[0], self.y_val[0], self.y_test[0]),
                                                  (self.y[-1], self.y_val[-1], self.y_test[-1]),
                                                  plot_names):
            out, _, _, out_aux = self.model.predict(inp, batch_size=512)
            for row, row_aux, row_out, row_aux_out, index in \
                    zip(outp, outp_aux, out, out_aux, xrange(len(outp))):
                if index % n == 0:
                    for ax, joint, joint_out, joint_name in \
                            zip(axs, np.append(row.T, row_aux.T, axis=0),
                                np.append(row_out.T, row_aux_out.T, axis=0), joint_names):
                        ax.clear()
                        ax.plot(joint)
                        ax.plot(joint_out)
                        ax.set_title(joint_name)

                    f.savefig(self.img_path + plot_name + str(index) + '.pdf', dpi='400')
        plt.close('all')


def parse():
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('-t', '--train', help='Begin training', action='store_true')
    parser.add_argument('-r', '--resume', help='Resume training', action='store_true')
    parser.add_argument('-v', '--visualization', help='Display predictions', action='store_true')
    parser.add_argument('filename', help='Name of the file to load')
    parser.add_argument('-n', '--nrollout', help='Number of rollouts', type=int)
    parser.add_argument('-e', '--epoch', help='Number of epoch', type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse()
    n_rollout = args.nrollout
    n_epoch = args.epoch

    seed = 1098
    np.random.seed(seed)
    path = args.filename
    names = ['target_pos', 'target_speed', 'pos', 'vel', 'effort']
    with h5py.File(path, 'r') as f:
        (target_pos, target_speed, pos, vel, effort) = [[np.array(val) for val in f[name_].values()] for name_ in names]

    x_target = np.array(target_pos)
    x_first = np.array([pos_[0] for pos_ in pos])
    x_speed = np.array(target_speed).reshape((-1, 1))
    aux_output = [np.ones(eff_.shape[0]).reshape((-1, 1)) for eff_ in effort]

    x = np.concatenate((x_target, x_first, x_speed), axis=1)

    def prepare_time_data(data):
        data_scaler = StandardScaler()
        data_concat = np.concatenate([data_ for data_ in data], axis=0)
        data_scaler.fit(data_concat)
        new_data = [data_scaler.transform(data_) for data_ in data]

        return data_scaler, new_data

    input_scaler = StandardScaler()
    x = input_scaler.fit_transform(x)
    effort_scaler, effort = prepare_time_data(effort)
    pos_scaler, pos = prepare_time_data(pos)
    vel_scaler, vel = prepare_time_data(vel)

    torque = pad_sequences(effort, padding='post', value=0., dtype=np.float64)
    pos = pad_sequences(pos, padding='post', value=0., dtype=np.float64)
    vel = pad_sequences(vel, padding='post', value=0., dtype=np.float64)
    aux_output = pad_sequences(aux_output, padding='post', value=0., dtype=np.float64)
    mask = aux_output[:, :, 0]
    aux_mask = np.ones(aux_output.shape[:2])

    x, x_test, torque, torque_test, pos, pos_test, vel, vel_test, \
    aux, aux_test, mask, mask_test, aux_mask, aux_mask_test = \
        train_test_split(x, torque, pos, vel, aux_output, mask, aux_mask, test_size=0.3, random_state=seed)

    kf = KFold(n_splits=3, shuffle=True, random_state=seed)

    if not os.path.exists('save_model_selection'):
        os.makedirs('save_model_selection')

    for (train_index, cv_index), i in zip(kf.split(x), range(kf.n_splits)):
        widths_gru = [1000, 100]
        depths_gru = [1, 2]
        names = ['gru:{}-{}_conv:False_fold:{}'.format(width_, depth_, i) for
                 width_, depth_ in zip(widths_gru, depths_gru)]
        save_names = ['save_model_selection/' + name_ for name_ in names]
        log_names = ['log_model_selection/' + name_ for name_ in names]
        img_names = ['imgs/' + name_ for name_ in names]

        this_x, this_torque, this_pos, this_vel, this_aux, this_mask, this_aux_mask = \
            [a_[train_index] for a_ in [x, torque, pos, vel, aux, mask, aux_mask]]
        this_x_cv, this_torque_cv, this_pos_cv, this_vel_cv, this_aux_cv, this_mask_cv, this_aux_mask_cv = \
            [a_[cv_index] for a_ in [x, torque, pos, vel, aux, mask, aux_mask]]

        for width_gru, depth_gru, save_name, log_name, img_name in \
                zip(widths_gru, depths_gru, save_names, log_names, img_names):
            model = MyModel(train=[this_x, [this_torque, this_pos, this_vel, this_aux]],
                            val=[this_x_cv, [this_torque_cv, this_pos_cv, this_vel_cv, this_aux_cv]],
                            train_mask=[this_mask] * 3 + [this_aux_mask],
                            val_mask=[this_mask_cv] * 3 + [this_aux_mask_cv],
                            test=[x_test, [torque_test, aux_test]], test_mask=[mask_test, aux_mask_test],
                            max_unroll=n_rollout, save_dir=save_name, log_dir=log_name, img_dir=img_name,
                            width_gru=width_gru, depth_gru=depth_gru, width_dense=50, depth_dense=2)
            if args.train:
                model.fit(nb_epoch=n_epoch, batch_size=512)
            elif args.resume:
                model.resume(nb_epoch=n_epoch, batch_size=512)
            elif args.visualization:
                model.load()
                model.save_figs()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

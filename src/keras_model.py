import pickle

import h5py
import numpy as np
import time
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers.convolutional import Convolution1D
from keras.layers.core import RepeatVector, Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
from sklearn.model_selection._split import train_test_split


def keras_model(max_unroll):
    model = Sequential()
    model.add(RepeatVector(max_unroll, input_shape=(15,)))
    model.add(TimeDistributed(Dense(64)))
    model.add(Dropout(0.1))
    model.add(LSTM(64, return_sequences=True, dropout_U=0.05, dropout_W=0.1))
    model.add(Convolution1D(50, 3, border_mode='same', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Convolution1D(20, 3, border_mode='same', activation='relu'))
    model.add(Dropout(0.1))
    model.add(TimeDistributed(Dense(50, activation='relu')))
    model.add(Dropout(0.1))
    model.add(TimeDistributed(Dense(7)))

    return model

path = '../DataBase/data.hdf5'
names = ['target_pos', 'target_speed', 'pos', 'vel', 'effort']
np.random.seed(1098)
data = list()
with h5py.File(path, 'r') as f:
    (target_pos, target_speed, pos, vel, effort) = [[np.array(val) for val in f[name].values()] for name in names]
# todo normalizar datos

x_target = np.array(target_pos)
x_first = np.array([pos_[0] for pos_ in pos])
x_speed = np.array(target_speed).reshape((-1, 1))
x = np.concatenate((x_target, x_first, x_speed), axis=1)

y = pad_sequences(effort, padding='post', value=1000.)
# plt.plot(y[0, :, :])
# plt.show()
x, x_test, y, y_test = train_test_split(x, y, test_size=0.2)
print(np.shape(y), np.shape(y_test))
kfold = KFold(n_splits=10, shuffle=True)
cvscores = []
mask = np.sum(y, axis=2) != 7000.
# print(np.sum(mask, axis=1))

times = [time.time()]
for i, (train, val) in enumerate(kfold.split(x, y)):
    model = keras_model(1502)

    model.compile(optimizer='adam', loss='mse', sample_weight_mode='temporal',
                  metrics=['accuracy', mean_squared_error, mean_absolute_error])

    saveCallback = ModelCheckpoint('save/' + str(i) + '-Fold-model_checkpoint.{epoch:03d}-{acc:.3f}.hdf5')
    tensorboardCallback = TensorBoard(histogram_freq=10)
    model.fit(x[train], y[train], nb_epoch=10, batch_size=32,
              callbacks=[saveCallback, tensorboardCallback], sample_weight=mask[train])

    scores = model.evaluate(x[val], y[val], sample_weight=mask[val])
    cvscores.append(scores)
    times.append(time.time())

with open('save/cvscores', 'w') as f:
    pickle.dump(cvscores, f)
    pickle.dump(times, f)



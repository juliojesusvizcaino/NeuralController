import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers.core import RepeatVector, Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from sklearn.model_selection import KFold


def keras_model(max_unroll):
    model = Sequential()
    model.add(RepeatVector(max_unroll, input_shape=(15)))
    model.add(LSTM(20, return_sequences=True, dropout_U=0.2, dropout_W=0.2))
    model.add(TimeDistributed(Dense(20, activation='relu')))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(7)))

    return model

kfold = KFold(n_splits=10, shuffle=True)
cvscores = []

for i, train, x_val in enumerate(kfold.split(x, y)):
    model = keras_model(1500)

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy', mean_squared_error, mean_absolute_error])

    saveCallback = ModelCheckpoint(str(i) + '-Fold-model_checkpoint.{epoch:03d}-{val_loss:.3f}.hdf5')
    tensorboardCallback = TensorBoard()
    model.fit(x[train], y[train], callbacks=[saveCallback, tensorboardCallback])


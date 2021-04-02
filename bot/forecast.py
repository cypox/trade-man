from datasource import datafeed
from dataset import dataset

import datetime
import keras
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Conv1D, concatenate, Flatten, BatchNormalization

class predictor:
  def __init__(self):
    self.ticker = None
    self.model = None
  
  def build_model(self, input_size = 30, num_outputs = 1):
    model_input = Input(shape=(input_size, 1))
    # the first branch operates on the first input
    x = BatchNormalization()(model_input)
    x = LSTM(32, return_sequences=True)(x)
    x = LSTM(32, return_sequences=True)(x)
    x = LSTM(16, return_sequences=True)(x)
    x = LSTM(16, return_sequences=False)(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(8, activation="relu")(x)
    x = Dense(num_outputs, activation="linear")(x)
    self.model = Model(inputs=model_input, outputs=x)
  
  def train_model(self, dataset, epochs=10):
    optimizer = Adam(lr=1e-3, decay=1e-3 / 200)
    self.model.compile(optimizer=optimizer , loss='mean_squared_error')
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    tensorboard_callback = []
    self.model.fit(dataset.train, validation_data=dataset.valid, epochs=epochs, callbacks=[tensorboard_callback])
  
  def predict_single(self, input_sequence):
    return self.model.predict([input_sequence])[0]

  def predict(self, input_sequences):
    return self.model.predict(input_sequence)
  
  def save_model(self, path):
    self.model.save(path)
  
  def load_model(self, path):
    self.model = keras.models.load_model(path)

load_saved = True
p = predictor()
if load_saved:
  p.load_model('trained.h5')
else:
  p.build_model()
  f = datafeed()
  f.update_ticker('NIO')
  d = dataset()
  d.build_dataset(f.get_dataframe())
  p.train_model(d)
  p.save_model('trained.h5')

import random
input_seq = [random.random() * 5 + 30 for i in range(30)]
expected_seq = random.random() * 5 + 30
output_seq = p.predict_single(input_seq)

from plot_utils import show_plot
plt = show_plot([input_seq, expected_seq, output_seq], 0, 'Sample Example')
plt.savefig('prediction_output.png')

from datasource import datafeed
from dataset import dataset
from predictor.lstm_forecast import predictor
from plot_utils import show_plot

import random
import math

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

f = datafeed()
f.update_ticker('NIO')
d = dataset()
d.build_dataset(f.get_dataframe())

output_seq = p.predict(d.valid)

error = 0.
global_idx = 0
for i in d.valid:
  index = 0
  for j in i[1]:
    diff = (j[0] - output_seq[global_idx][0])
    error += diff * diff
    global_idx += 1
    index += 1
    if index == 32:
      index = 0

print('mse = {}'.format(math.sqrt(error)/global_idx))

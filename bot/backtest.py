from datasource import datafeed
from dataset import dataset
from predictor.lstm_forecast import predictor
from plot_utils import show_plot

import random

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
f.update_ticker('AAPL')
d = dataset()
d.build_dataset(f.get_dataframe())

output_seq = p.predict(d.valid)

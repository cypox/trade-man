import pandas as pd
from pandas_datareader.av.time_series import AVTimeSeriesReader
import matplotlib.pyplot as plt

class datafeed:
  def __init__(self):
    self.API_KEY = '1WVZZXH3ZU1GRKQR'
    self.function = 'TIME_SERIES_INTRADAY'
    self.symbol = None
    self.interval = '1min'
    self.data = None
  
  def update_ticker(self, ticker, update_dataframe=True):
    self.symbol = ticker
    if update_dataframe:
      self.data = None
      self.update_dataframe()

  def get_dataframe(self):
    return self.data

  def update_dataframe(self):
    if self.symbol is None:
      return
    if self.data is None:
      self.data = AVTimeSeriesReader(symbols=self.symbol, function=self.function, api_key=self.API_KEY).read()
    else:
      last_timestamp = self.data.index[-1]
      new_data = AVTimeSeriesReader(symbols=self.symbol, start=last_timestamp, function=self.function, api_key=self.API_KEY).read()
      if len(new_data) > 0:
        frames = [self.data, new_data]
        self.data = pd.concat(frames).drop_duplicates()

  def plot(self, columns=['close']):
    self.data[columns].plot()
    plt.savefig('{}.png'.format(self.symbol))

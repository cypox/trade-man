import tensorflow as tf

class dataset:
  def __init__(self):
    self.train = None
    self.valid = None
    self.batch_size = 32
    self.SHUFFLE_BUFFER_SIZE = 10000
    self.size = 0
  
  def build_dataset(self, data, input_sequence_length = 30, output_sequence_length = 1):
    close_prices = data['close']
    num_sequences = len(close_prices) - input_sequence_length - output_sequence_length + 1
    if num_sequences < 0:
      return None
    sequences = []
    targets = []
    for i in range(num_sequences):
      input_seq = close_prices[i:i+input_sequence_length].values
      output_seq = close_prices[i+input_sequence_length:i+input_sequence_length+output_sequence_length].values
      sequences.append(input_seq)
      targets.append(output_seq)
    dataset = tf.data.Dataset.from_tensor_slices((sequences, targets))
    self.size = len(dataset)
    print('database created with {} entries.'.format(self.size))
    dataset = dataset.shuffle(self.SHUFFLE_BUFFER_SIZE)
    self.train_size = int(self.size * 0.7)
    self.valid_size = self.size - self.train_size

    train_dataset = dataset.take(self.train_size)
    valid_dataset = dataset.skip(self.train_size)
    valid_dataset = dataset.take(self.valid_size)

    self.train = train_dataset.batch(self.batch_size)
    self.valid = valid_dataset.batch(self.batch_size)

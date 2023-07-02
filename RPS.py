import math
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def player(prev_play, opponent_history=[], prev_history=[], prediction_data=[]):
    test = True
    if prev_play != '':
      opponent_history.append(prev_play)

    min_random_play = 10
    correct_guess = 0
    highest_accuracy = 0

    early_stopping = EarlyStopping()

    data = pd.Series(opponent_history)
    data_u = pd.Series(prev_history)

    if data.shape[0] > min_random_play:
      input_size = len(data)
      vocab = ["R", "P", "S"]
      layer = keras.layers.StringLookup(vocabulary=vocab)
      data_u = layer(data)
      data_u = data_u - 1
      seq_length = round(input_size*0.6)

      dataset = keras.utils.timeseries_dataset_from_array(
          data_u.numpy(),
          targets = data_u[seq_length:],
          sequence_length = seq_length,
          batch_size = 1000
      )

      model = keras.Sequential([
          keras.layers.SimpleRNN(32, activation='relu', input_shape=[None,1]),
          keras.layers.Dense(3, activation='softmax')
      ])

      model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

      history = model.fit(dataset, epochs=500, shuffle=True, verbose=False, callbacks=[early_stopping])
      accuracy = history.history['accuracy'][-1]
      actual_model = model

      if accuracy >= highest_accuracy:
          highest_accuracy = accuracy
          best_model = actual_model
          best_seq = [seq_length, history.epoch[-1]]
        
      if prev_history != []:
          for i in range(0,len(prev_history)):
            if opponent_history[-i-1] == prev_history[-i-1]:
                correct_guess += 1
            if len(prev_history) > 0:
                correct_guess_rate = correct_guess/(len(prev_history))

      x_pred = data_u[-seq_length:]
      x_pred = tf.reshape(x_pred, [1,seq_length])
      predchance = best_model.predict(x_pred)
      max_n = np.argmax(predchance)
      pred = layer.get_vocabulary()[max_n+1]
      prev_history.append(pred)

      if pred == 'R':
        guess = 'P'
      elif pred == 'P':
        guess = 'S'
      else:
        guess = 'R'
    else:
       guess = random.choice(['R','P','S'])
       prev_history.append(guess)
    return guess

class EarlyStopping(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') > 0.8):
      self.model.stop_training = True

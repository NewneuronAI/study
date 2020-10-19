
from pathlib import Path
import time
import os
from scipy.io import wavfile
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import keras

from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Input, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


def get_data(path):
    ''' Returns dataframe with columns: 'path', 'word'.'''
    datadir = Path(path)
    files = [(str(f), f.parts[-2]) for f in datadir.glob('**/*.wav') if f]
    df = pd.DataFrame(files, columns=['path', 'word'])

    return df


def prepare_data(df):
    '''Transform data into something more useful.'''
    train_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    words = df.word.unique().tolist()
    silence = ['_background_noise_']
    unknown = [w for w in words if w not in silence + train_words]

    # there are only 6 silence files. Mark them as unknown too.
    df.loc[df.word.isin(silence), 'word'] = 'unknown'
    df.loc[df.word.isin(unknown), 'word'] = 'unknown'

    return df
test = prepare_data(get_data('D:/s/Tensorflowspeechrecognition/test/test/'))

predictions = []
paths = test.path.tolist()
x_test = open('D://s/Tensorflowspeechrecognition/list.txt','r')
t =x_test.readlines()
labels=[]
for i in t:
    labels.append(str(i.rstrip()))
print(labels)
fname = test.path.tolist()

a=[]

for i in fname:
    i = os.path.split(i)
    a.append(i[-1])

fname = a
print(fname)
submission = pd.DataFrame({'fname': fname, 'label': labels})
submission.to_csv('D:/s/Tensorflowspeechrecognition/submission-end.csv', index=False)
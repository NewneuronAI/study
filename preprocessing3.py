import os
import librosa
import numpy as np
from glob import glob
from tqdm import tqdm
Audio = 'D:/s/Tensorflowspeechrecognition/train/train/audio'
Audio_list = os.listdir(Audio)

import soundfile as sf
def shifting(data, sampling_rate, shift_max, shift_direction):
    shift = np.random.randint(sampling_rate * shift_max+1)
    if shift_direction == 'right':
       shift = -shift
    elif shift_direction == 'both':
       direction = np.random.randint(0, 2)
       if direction == 1:
          shift = -shift
    augmented_data = np.roll(data, shift)
# Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data

def change_pitch(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

# def split_arr(arr):
#     """
#     split an array into chunks of length 16000
#     Returns:
#         list of arrays
#     """
#     return np.split(arr, np.arange(0, 16000, 4000))

L1 = list()
L2 = list()
L3 = list()
L4 = list()
train_dir = 'D:/s/Tensorflowspeechrecognition/train/train/audio' #download files from kaggle
import csv
with open('D:/s/Tensorflowspeechrecognition/train/train/validation_list.txt') as val_list:
    validation_list = [row[0] for row in csv.reader(val_list)]
assert len(validation_list) == 6798, 'file not loaded'
#validation list 가져오기
#validation_list : silence만 없음


training_list = []

for file in Audio_list:

    for wavs in os.listdir(Audio+'/'+file):

        training_list.append(Audio+'/'+file+'/'+wavs)


#
import random
# # print(training_list)

for i in tqdm(range(4000)):
    data = random.choice(training_list)
    try:
        sig, sr = librosa.load(data, sr=16000)
        sig = np.array(sig)
        L1.append(sig[:4000])
        L2.append(sig[4000:8000])
        L3.append(sig[8000:12000])
        L4.append(sig[12000:16000])

    except:
        continue

from scipy.io.wavfile import write
def split_arr1(arr):
    return np.split(arr, np.arange(16000, len(arr), 16000))

#
# L1 =split_arr1(L1)

L1 = np.array(L1)
print(L1.shape)
L2 = np.array(L2)
print(L2.shape)
L3 = np.array(L3)
print(L3.shape)

L4 = np.array(L4)
print(L4.shape)
L = np.concatenate([L1,L2,L3,L4],axis=-1)


for index, data in enumerate(tqdm(L)):
    filename = 'frag%d' % (index) + 'L'
    write('D:/s/Tensorflowspeechrecognition/train/train/audio/unknown/' + filename + '.wav', 16000, data)

#
# L2 =split_arr1(L2)
#
# for index, data in enumerate(tqdm(L2)):
#     filename = 'frag%d' % (index) + 'L2'
#     write('D:/s/Tensorflowspeechrecognition/train/train/audio/unknown/' + filename+'.wav',16000, data)
#
# L3 = split_arr1(L3)
#
# for index, data in enumerate(tqdm(L3)):
#     filename = 'frag%d' % (index) + 'L3'
#     write('D:/s/Tensorflowspeechrecognition/train/train/audio/unknown/' + filename+'.wav',16000, data)
#
# L4 = split_arr1(L4)
#
# for index, data in enumerate(tqdm(L4)):
#     filename = 'frag%d' % (index) + 'L4'
#     write('D:/s/Tensorflowspeechrecognition/train/train/audio/unknown/' + filename+'.wav',16000, data)
# # #
# #



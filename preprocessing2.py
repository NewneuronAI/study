import numpy as np
import librosa
import librosa.display
import os
import csv
import soundfile as sf

import matplotlib.pyplot as plt
import random

train_dir = 'D:/s/Tensorflowspeechrecognition/train/train/audio' #download files from kaggle

classes = ['yes', 'no',
           'up', 'down',
           'left', 'right',
           'on', 'off',
           'stop', 'go',
           'silence', 'unknown']

def split_arr(arr):
    """
    split an array into chunks of length 16000
    Returns:
        list of arrays
    """
    return np.split(arr, np.arange(16000, len(arr), 16000))

#
import librosa
import soundfile as sf
from glob import glob
from IPython.display import Audio
import IPython.display as ipd
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
# import mdp
import tensorflow as tf
import os
import tensorflow_io as tfio

import tensorflow.keras.backend as K
def noising(data,noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
# Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

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
def create_silence():
    """
    reads wav files in background noises folder,
    splits them and saves to silence folder in train_dir
    """
    for file in os.listdir('D:/s/Tensorflowspeechrecognition/train/train/_background_noise_/'):
        if 'wav' in file:
            sig, rate = librosa.load('D:/s/Tensorflowspeechrecognition/train/train/_background_noise_/' + file, sr = 16000)
            sig = shifting(sig, rate, 0.5, 'both')
            sig = change_pitch(sig,rate,np.random.randint(-10,10))
            noising(sig,np.random.randint(10))
            sig_arr = split_arr(sig)


            if not os.path.exists(train_dir+'silence5/'):
                os.makedirs(train_dir+'silence5/')
            for ind, arr in enumerate(sig_arr):
                filename = 'frag%d' %(ind+384) + '_%s' %file # example: frag0_running_tap.wav
                sf.write(train_dir+'silence5/'+filename, arr, 16000)



            #     librosa.output.write_wav(train_dir+'silence/'+filename, arr, 16000)

create_silence()

#
#
# folders = os.listdir(train_dir)
# # put folders in same order as in the classes list, used when making sets
# all_classes = [x for x in classes[:11]]
#
# for ind, cl in enumerate(folders):
#     if cl not in classes:
#         all_classes.append(cl) #모든 종류의 클래스 리스트
#
#
# with open('D:/s/Tensorflowspeechrecognition/train/train/validation_list.txt') as val_list:
#     validation_list = [row[0] for row in csv.reader(val_list)]
# assert len(validation_list) == 6798, 'file not loaded'
# #validation list 가져오기
# #validation_list : silence만 없음
#
# for i, file in enumerate(os.listdir(train_dir + '/silence/')):
#     if i%10==0:
#         validation_list.append('silence/'+file)
# # print(validation_list)
# #silence 추가
# training_list = []
# all_files_list = []
# class_counts = {}
#
# for folder in folders:
#     files = os.listdir(train_dir+'/' + folder)
#     for i, f in enumerate(files):
#         all_files_list.append(folder + '/' + f)
#         path = folder + '/' + f
#         if path not in validation_list:
#             training_list.append(folder + '/' + f)
#         class_counts[folder] = i
#
# # print(training_list)
# # print(all_files_list)
# # print(class_counts)  #총개수
# validation_list = list(set(validation_list).intersection(all_files_list))
# # print(validation_list)
# assert len(validation_list)+len(training_list)==len(all_files_list), 'error'
# #
# # #
# # #
# # # x, r = librosa.load(train_dir + 'yes/bfdb9801_nohash_0.wav', sr = 16000)
# # #
# #
#
# def make_spec(file, file_dir=train_dir, flip=False, ps=False, st=4):
#
#
#     sig, rate = librosa.load(file_dir +'/'+file, sr=16000)
#     if len(sig) < 16000:  # pad shorter than 1 sec audio with ramp to zero
#         sig = np.pad(sig, (0, 16000 - len(sig)), 'linear_ramp')
#     S = librosa.feature.melspectrogram(y=sig, n_mels=128, n_fft=1024, hop_length=128, fmax=8000, center=True,
#                                        window='bartlett')
#     S = librosa.amplitude_to_db(S, ref=np.max)
#     # print(S.shape)
#     return S.astype(np.float32)
# # # # #
# # # # librosa.display.specshow(make_spec('yes/bfdb9801_nohash_0.wav'),
# # # #                          x_axis='mel',
# # # #                          fmax=8000,
# # # #                          y_axis='time',
# # # #                          sr = 16000,
# # # #                          hop_length = 128)
# # # #
# # #
# #
# def create_sets(file_list=training_list):
#     X_array = np.zeros([len(file_list), 128, 126])
#     Y_array = np.zeros([len(file_list)])
#     for ind, file in enumerate(file_list):
#         if ind % 2000 == 0:
#             print(ind, file)
#         try:
#             X_array[ind] = make_spec(file)
#         except ValueError:
#             print(ind, file, ValueError)
#         Y_array[ind] = all_classes.index(file.rsplit('/')[0])
#
#     return X_array, Y_array
# x,y = create_sets()
# # print(y)
# X_train, Y_train_all = create_sets() # takes a while
#
# Y_train = np.where(Y_train_all < 11, Y_train_all, 11)
# #
#
#
# np.save('D:/s/Tensorflowspeechrecognition/X_train4b.npy', np.expand_dims(X_train, -1)+1.3)
# # np.save('D:/s/Tensorflowspeechrecognition/Y_train2.npy', Y_train.astype(np.int))
# # np.save('D:/s/Tensorflowspeechrecognition/Y_train_all.npy', Y_train_all.astype(np.int))
#
#
# X_val, Y_val_all = create_sets(file_list = validation_list)
#
# Y_val = np.where(Y_val_all < 11, Y_val_all, 11)
#
# np.save('D:/s/Tensorflowspeechrecognition/X_val4b.npy', np.expand_dims(X_val, -1)+1.3)
# # np.save('D:/s/Tensorflowspeechrecognition/Y_val2.npy', Y_val.astype(np.int))
# # np.save('D:/s/Tensorflowspeechrecognition/Y_val_all.npy', Y_val_all.astype(np.int))

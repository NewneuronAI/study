import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.utils import to_categorical, plot_model
from keras import backend as K
from time import time
from resnst import ResNet
from collections import Counter
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from sklearn.model_selection import train_test_split
X_train = np.load('D:/s/Tensorflowspeechrecognition/X_train.npy')
Y_train = np.load('D:/s/Tensorflowspeechrecognition/Y_train.npy')
X_val = np.load('D:/s/Tensorflowspeechrecognition/X_val.npy')
Y_val = np.load('D:/s/Tensorflowspeechrecognition/Y_val.npy')

X_test,X_val,Y_test,Y_val = train_test_split(X_val,Y_val,train_size=0.5)


classes = ['yes', 'no',
           'up', 'down',
           'left', 'right',
           'on', 'off',
           'stop', 'go',
           'silence', 'unknown']

all_classes = [x for x in classes[:11]]
pred_class ={0:'yes', 1:'no', 2:'up', 3:'down', 4:'left', 5:'right', 6:'on', 7:'off', 8:'stop', 9:'go',10:'silence',
             11:'unknown'}


for ind, cl in enumerate(os.listdir('D:/s/Tensorflowspeechrecognition/train/train/audio/')):
    if cl not in classes:
        all_classes.append(cl)

def get_class_weights(y):
    counter = Counter(y)

    majority = max(counter.values())

    return  {cls: float(majority/count) for cls, count in counter.items()}

class_weights = get_class_weights(Y_train)

input_size = X_train.shape[1:]

batch_size = 196

filters_list = [8,16,32]
output_size = 12

date = '1003'
arch = 'resnet8_16_32'
import tensorflow as tf
from tensorflow.keras.layers import *


sr = ResNet(filters_list, input_size, output_size)
sr.build()

sr.m.compile(loss='categorical_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])
# sr.m.load_weights('D:/s/Tensorflowspeechrecognition/models/Epoch_22_Val_0.227best.h5',by_name=True)
# sr.m.trainable=False
# # plot_model(sr.m,
# #            to_file = './models/{}_{}.png'.format(arch,date),
# #            show_shapes = True)
#
# X_test = np.load('D:/s/Tensorflowspeechrecognition/X_test.npy',allow_pickle=True)
# import dask.array as da
# X_test = da.array(X_test)
# pred = sr.m.predict(X_test)
# p=[]
# for i in pred:
#     index = np.argmax(i)
#     a=pred_class[index]
#     p.append(a)
# fip = open('D://s/Tensorflowspeechrecognition/list.txt','w')
# fip.write('\n'.join(p))
# fip.close()
#


checkpointer = ModelCheckpoint(filepath='D:/s/Tensorflowspeechrecognition/models/Epoch_{epoch:02d}_Val_{val_loss:.3f}best.h5',
                               verbose=0,
                               save_weights_only=True)


history = sr.m.fit(X_train,
                   to_categorical(Y_train),
                   batch_size = batch_size,
                   epochs = 1500,
                   verbose = 1, shuffle = True,
                   class_weight = class_weights,
                   validation_data = (X_val, to_categorical(Y_val)),
                   callbacks = [checkpointer]) # add more callbacks if you want

sr.m.eveluate(X_test,to_categorical(Y_test))
# sr.m.save_weights("./models/{}_{}_last.h5".format(arch, date))



#
#
# #

#
# from resnst import CTC, ctc_lambda_func #used in the CTC build method
# from ctc_params import char_map, index_map, text_to_int, get_intseq, get_ctc_params
#
# def ctc(y_true, y_pred):
#     return y_pred
#
# sr_ctc = CTC((122,85), 28)
# sr_ctc.build()
#
#
# sr_ctc.m.compile(loss = ctc, optimizer = 'adam', metrics = ['accuracy'])
# sr_ctc.tm.compile(loss = ctc, optimizer = 'adam')
#
#
#
# Y_train_all = np.load('D:/s/Tensorflowspeechrecognition/Y_train_all.npy')
# Y_val_all = np.load('D:/s/Tensorflowspeechrecognition/Y_val_all.npy')
#
# labels, input_length, label_length = get_ctc_params(Y = Y_train_all, classes_list = all_classes)
# labels_val, input_length_val, label_length_val = get_ctc_params(Y = Y_val_all, classes_list = all_classes)
#
#
# checkpointer = ModelCheckpoint(filepath="./models/ctc_{}_best.h5".format(date),
#                                verbose=0,
#                                save_best_only=True)
#
#
#
# history = sr_ctc.m.fit([np.squeeze(X_train),
#                             labels,
#                             input_length,
#                             label_length],
#                        np.zeros([len(Y_train_all)]),
#                        batch_size = 128,
#                        epochs = 10,
#                        validation_data = ([np.squeeze(X_val),
#                                            labels_val,
#                                            input_length_val,
#                                            label_length_val],
#                                           np.zeros([len(Y_val_all)])),
#                        callbacks = [checkpointer],
#                        verbose = 1, shuffle = True)
# #
# # sr_ctc.m.save_weights('./models/ctc_{}.h5'.format(date))
# # sr_ctc.tm.load_weights('./models/ctc_{}_best.h5'.format(date))
# X_test = np.load('D:/s/Tensorflowspeechrecognition/X_test.npy',allow_pickle=True)
# print(X_test.shape)
# #Xtest
# def str_out(dataset=X_test):
#     k_ctc_out = K.ctc_decode(sr_ctc.tm.predict(np.squeeze(dataset),
#                                                verbose=1),
#                              np.array([28 for _ in dataset]))
#     decoded_out = K.eval(k_ctc_out[0][0])
#     str_decoded_out = []
#     for i, _ in enumerate(decoded_out):
#         str_decoded_out.append("".join([index_map[c] for c in decoded_out[i] if not c == -1]))
#
#     return str_decoded_out
#
# y_pred_val = str_out()
# z= []
# for i in y_pred_val:
#     if i == 'yes':
#         z.append(i)
#     elif i == 'no':
#         z.append(i)
#     elif i == 'up':
#         z.append(i)
#     elif i == 'down':
#         z.append(i)
#     elif i == 'left':
#         z.append(i)
#     elif i == 'right':
#         z.append(i)
#     elif i == 'on':
#         z.append(i)
#     elif i == 'off':
#         z.append(i)
#     elif i == 'stop':
#         z.append(i)
#     elif i == 'go':
#         z.append(i)
#     elif i == 'silence':
#         z.append(i)
#     elif i == 'unknown':
#         z.append(i)
#     else:
#         i = 'unknown'
#         z.append(i)
#
# fip = open('D://s/Tensorflowspeechrecognition/list.txt','w')
# fip.write('\n'.join(z))
# fip.close()
#






# print('PREDICTED: \t REAL:')
# for i in range(10):
#     print(y_pred_val[i], '\t\t',all_classes[Y_val_all[i]])

# print(classification_report([all_classes[Y_val_all[i]] for i, _ in enumerate(Y_val_all)],
#                             y_pred_val, labels = all_classes))

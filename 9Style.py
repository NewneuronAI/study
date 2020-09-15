import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.layers import MaxPooling2D,MaxPooling3D,AveragePooling3D,Conv3D,Bidirectional,Attention,GRU,TimeDistributed,LSTM
from tensorflow.keras.layers import Dense,Activation,TimeDistributed,Flatten,Conv2D,BatchNormalization,AveragePooling2D,Dropout
from tensorflow.keras.models import Model,Sequential
# from sklearn.model_selection import train_test_split
import os
import os.path as pth
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer,Input,GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from glob import glob
from activation import mish
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import LearningRateScheduler, CSVLogger, ModelCheckpoint
import tensorflow.keras.backend as K
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
import tensorflow as tf
from config import NUM_CLASSES
from models.residual_block import make_basic_block_layer, make_bottleneck_layer
import tensorflow.keras.backend as k
from Convpractice import *
import dask.array as da
# from sklearn.model_selection import train_test_split,KFold
from dask_ml.model_selection import train_test_split,KFold
# from sklearn.model_selection import KFold
from RAdam import RAdam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#-----------------------------------------------------------------------------------------------------------------------
# def data_loader(files):
#     out=[]
#     for file in range(files):
#         data,sr=librosa.load(file,sr=16000)
#         data=librosa.feature.melspectrogram(data,sr=sr,n_mels=32, n_fft=1024, hop_length=512,fmin=30,fmax=8000,pad_mode='wrap',window=('kaiser',5.5))
#         log_S = librosa.amplitude_to_db(data, ref=np.max)
#         out.append(log_S)
#     out=np.array(out)
#     return out
#Load-------------------------------------------------------------------------------------------------------------------
# X_data1 = np.load('D:/GAT/Sound/train1.npy')
# X_data1 = da.array(X_data1)
# X_data2 = np.load('D:/GAT/Sound/train2.npy')
# X_data2 = da.array(X_data2)
# X_data3 = np.load('D:/GAT/Sound/train3.npy')
# X_data3 = da.array(X_data3)
# X_data4 = np.load('D:/GAT/Sound/train4.npy')
# X_data4 = da.array(X_data4)
# X_data5 = np.load('D:/GAT/Sound/train5.npy')
# X_data5 = da.array(X_data5)
#
#
#
# x_data = da.stack([X_data1,X_data2,X_data3,X_data4,X_data5,
#                          ],axis=-1)

#
X_data1 =np.load('D:/GAT/Sound/1next2b+.npy')
x_data1 = da.array(X_data1)
X_data2 =np.load('D:/GAT/Sound/2next2b+.npy')
x_data2 = da.array(X_data2)
X_data3 =np.load('D:/GAT/Sound/3next2b+.npy')
x_data3 = da.array(X_data3)

x_data=da.concatenate([x_data1,x_data2,x_data3],axis=-1)
print(x_data.shape)

y_data=pd.read_csv('D:/GAT/subm/train_answer.csv', index_col=0)
y_labels = y_data.columns.values
y_data=y_data.values
Y_data=y_data

# #Preprocessing----------------------------------------------------------------------------------------------------------
x_train,xtest,y_train,ytest=train_test_split(x_data,Y_data,train_size=0.8,random_state=42)
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,train_size=0.8,random_state=42)

# kf = KFold(n_splits=4)
# for train_index, test_index in kf.split(x_data):
#     x_train, x_test = x_data[train_index], x_data[test_index]
#     y_train, y_test = y_data[train_index], y_data[test_index]
batch_size=32
# train_generator = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.1)
# train_Iterator = train_generator.flow(x_train, y_train,batch_size=batch_size)
#
# valid_generator = ImageDataGenerator()
# valid_Iterator = valid_generator.flow(x_test, y_test,batch_size=8)

input=(x_train.shape[1],x_train.shape[2],x_train.shape[3])

strategy=tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

def Jenson_Shannon_loss(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    y_true /= k.sum(y_true)
    y_pred /= k.sum(y_pred)
    m = (y_true + y_pred) / 2

    return (k.sum(y_true * k.log(y_true / m),axis=-1) + k.sum(y_pred * k.log(y_pred / m),axis=-1)) / 2

def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)

class MaxBlurPooling2D(Layer):

    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        self.pool_size = pool_size
        self.blur_kernel = None
        self.kernel_size = kernel_size

        super(MaxBlurPooling2D, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.kernel_size == 3:
            bk = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]])
            bk = bk / np.sum(bk)
        elif self.kernel_size == 5:
            bk = np.array([[1, 4, 6, 4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1, 4, 6, 4, 1]])
            bk = bk / np.sum(bk)
        else:
            raise ValueError

        bk = np.repeat(bk, input_shape[3])
        bk = da.array(bk)

        bk = da.reshape(bk, (self.kernel_size, self.kernel_size, input_shape[3], 1))
        blur_init = tf.keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(self.kernel_size, self.kernel_size, input_shape[3], 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(MaxBlurPooling2D, self).build(input_shape)

    def call(self, x):

        x = tf.nn.pool(x, (self.pool_size, self.pool_size),
                       strides=(1, 1), padding='SAME', pooling_type='MAX', data_format='NHWC')
        x = K.depthwise_conv2d(x, self.blur_kernel, padding='same', strides=(self.pool_size, self.pool_size))

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.ceil(input_shape[1] / 2)), int(np.ceil(input_shape[2] / 2)), input_shape[3]
###################################################################################################
from tensorflow.python.keras import backend as K


def _bernoulli(shape, mean):
    return tf.nn.relu(tf.sign(mean - tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)))


class DropBlock2D(tf.keras.layers.Layer):
    def __init__(self, keep_prob, block_size, scale=True, **kwargs):
        super(DropBlock2D, self).__init__(**kwargs)
        self.keep_prob = float(keep_prob) if isinstance(keep_prob, int) else keep_prob
        self.block_size = int(block_size)
        self.scale = tf.constant(scale, dtype=tf.bool) if isinstance(scale, bool) else scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 4
        _, self.h, self.w, self.channel = input_shape.as_list()
        # pad the mask
        p1 = (self.block_size - 1) // 2
        p0 = (self.block_size - 1) - p1
        self.padding = [[0, 0], [p0, p1], [p0, p1], [0, 0]]
        self.set_keep_prob()
        super(DropBlock2D, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        def drop():
            mask = self._create_mask(tf.shape(inputs))
            output = inputs * mask
            output = tf.cond(self.scale,
                             true_fn=lambda: output * tf.cast(tf.size(mask),dtype=tf.float32) / tf.reduce_sum(mask),
                             false_fn=lambda: output)
            return output

        if training is None:
            training = K.learning_phase()
        output = tf.cond(tf.logical_or(tf.logical_not(training), tf.equal(self.keep_prob, 1.0)),
                         true_fn=lambda: inputs,
                         false_fn=drop)
        return output

    def set_keep_prob(self, keep_prob=None):
        """This method only supports Eager Execution"""
        if keep_prob is not None:
            self.keep_prob = keep_prob
        w, h = tf.cast(self.w,dtype=tf.float32), tf.cast(self.h,dtype=tf.float32)
        self.gamma = (1. - self.keep_prob) * (w * h) / (self.block_size ** 2) / \
                     ((w - self.block_size + 1) * (h - self.block_size + 1))

    def _create_mask(self, input_shape):
        sampling_mask_shape = tf.stack([input_shape[0],
                                       self.h - self.block_size + 1,
                                       self.w - self.block_size + 1,
                                       self.channel])
        mask = _bernoulli(sampling_mask_shape, self.gamma)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [1, 1, 1, 1], 'SAME')
        mask = 1 - mask
        return mask


class DropBlock3D(tf.keras.layers.Layer):
    def __init__(self, keep_prob, block_size, scale=True, **kwargs):
        super(DropBlock3D, self).__init__(**kwargs)
        self.keep_prob = float(keep_prob) if isinstance(keep_prob, int) else keep_prob
        self.block_size = int(block_size)
        self.scale = tf.constant(scale, dtype=tf.bool) if isinstance(scale, bool) else scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 5
        _, self.d, self.h, self.w, self.channel = input_shape.as_list()
        # pad the mask
        p1 = (self.block_size - 1) // 2
        p0= (self.block_size - 1) - p1
        self.padding = [[0, 0], [p0, p1], [p0, p1], [p0, p1], [0, 0]]
        self.set_keep_prob()
        super(DropBlock3D, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        def drop():
            mask = self._create_mask(tf.shape(inputs))
            output = inputs * mask
            output = tf.cond(self.scale,
                             true_fn=lambda: output * tf.cast(tf.size(mask),dtype=tf.float32) / tf.reduce_sum(mask),
                             false_fn=lambda: output)
            return output

        if training is None:
            training = K.learning_phase()
        output = tf.cond(tf.logical_or(tf.logical_not(training), tf.equal(self.keep_prob, 1.0)),
                         true_fn=lambda: inputs,
                         false_fn=drop)
        return output

    def set_keep_prob(self, keep_prob=None):
        """This method only supports Eager Execution"""
        if keep_prob is not None:
            self.keep_prob = keep_prob
        d, w, h = tf.cast(self.d,dtype=tf.float32), tf.cast(self.w,dtype=tf.float32), tf.cast(self.h,dtype=tf.float32)
        self.gamma = ((1. - self.keep_prob) * (d * w * h) / (self.block_size ** 3) /
                      ((d - self.block_size + 1) * (w - self.block_size + 1) * (h - self.block_size + 1)))

    def _create_mask(self, input_shape):
        sampling_mask_shape = tf.stack([input_shape[0],
                                        self.d - self.block_size + 1,
                                        self.h - self.block_size + 1,
                                        self.w - self.block_size + 1,
                                        self.channel])
        mask = _bernoulli(sampling_mask_shape, self.gamma)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool3d(mask, [1, self.block_size, self.block_size, self.block_size, 1], [1, 1, 1, 1, 1], 'SAME')
        mask = 1 - mask
        return mask
###################################################################################################
#Class------------------------------------------------------
class Stem(tf.keras.Model):
    def __init__(self):
        super(Stem,self).__init__()
        self.c1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='valid',
                         kernel_initializer='he_uniform',bias_initializer=None )
        self.b = BatchNormalization()
        self.c2 = Activation(mish)
        self.c3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                         kernel_initializer='he_uniform',bias_initializer=None)
        self.b1 = BatchNormalization()
        self.c4 = Activation(mish)
        self.Stem3_1 = MaxPooling2D((3, 3), strides=(2, 2))  # c4
        self.Stem3_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid',kernel_initializer='he_uniform',bias_initializer=None)  # c4
        self.b2 = BatchNormalization()
        self.Stem3_21 =Activation('relu')

        # self.Stem4 = Concatenate([self.Stem3_1, self.Stem3_2], axis=-1)

        self.Stem5 = Conv2D(filters=32, kernel_size=(1, 1), padding='same',kernel_initializer='he_uniform',bias_initializer=None)  # Stem4
        self.b3 = BatchNormalization()
        self.Stem5_1 = Activation(mish)
        self.Stem5_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same',kernel_initializer='he_uniform',bias_initializer=None)
        self.b4 = BatchNormalization()
        self.Stem5_3 = Activation(mish)

        self.Stem6 = Conv2D(filters=32, kernel_size=(1, 1), padding='same',kernel_initializer='he_uniform',bias_initializer=None)  # Stem4
        self.b5 = BatchNormalization()
        self.Stem6_1 = Activation(mish)
        self.Stem6_2 = Conv2D(filters=32, kernel_size=(7, 1), padding='same',kernel_initializer='he_uniform',bias_initializer=None)
        self.b6 = BatchNormalization()
        self.Stem6_3 = Activation(mish)
        self.Stem6_4 = Conv2D(filters=32, kernel_size=(1, 7), padding='same',kernel_initializer='he_uniform',bias_initializer=None)
        self.b7 = BatchNormalization()
        self.Stem6_5 = Activation(mish)
        self.Stem6_6 = Conv2D(filters=64, kernel_size=(3, 3), padding='same',kernel_initializer='he_uniform',bias_initializer=None)
        self.b8 = BatchNormalization()
        self.Stem6_7 = Activation(mish)

        # self.c5=concatenate([self.Stem6_7, self.Stem5_3], axis=-1)

        self.Stem7_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu',kernel_initializer='he_uniform',bias_initializer=None)  # c5
        self.Stem7_2 = MaxPooling2D((3, 3), strides=(2, 2))  # c5

        # self.c6=concatenate([self.Stem7_1,self.Stem7_2],axis=-1)

        self.bn1 = BatchNormalization()
        self.r1 = Activation('relu')

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b(x)
        x = self.c2(x)
        x = self.c3(x)
        x= self.b1(x)
        x = self.c4(x)

        s1 = self.Stem3_1(x)
        s2 = self.Stem3_2(x)
        s2 = self.b2(s2)
        s2 = self.Stem3_21(s2)

        x = tf.keras.layers.concatenate([s1, s2], axis=-1)

        s1 = self.Stem5(x)
        s1 = self.b3(s1)
        s1 = self.Stem5_1(s1)
        s1 = self.Stem5_2(s1)
        s1 = self.b4(s1)
        s1 = self.Stem5_3(s1)

        s2 = self.Stem6(x)
        s2 = self.b5(s2)
        s2 = self.Stem6_1(s2)
        s2 = self.Stem6_2(s2)
        s2 = self.b6(s2)
        s2 = self.Stem6_3(s2)
        s2 = self.Stem6_4(s2)
        s2 = self.b7(s2)
        s2 = self.Stem6_5(s2)
        s2 = self.Stem6_6(s2)
        s2 = self.b8(s2)
        s2 = self.Stem6_7(s2)

        x = tf.keras.layers.concatenate([s1, s2], axis=-1)

        s1 = self.Stem7_1(x)

        s2 = self.Stem7_2(x)

        x = tf.keras.layers.concatenate([s1, s2], axis=-1)

        x = self.bn1(x)

        x = self.r1(x)

        return x


class ct(tf.keras.Model):
    def __init__(self,filter):
        super(ct,self).__init__()

        self.conv1 = Conv2D(filters=filter, kernel_size=(1,1), padding='same', strides=(1,1), bias_initializer=None,kernel_initializer='he_uniform')
        self.bn = BatchNormalization()
        self.drop = DropBlock2D(keep_prob=0.8, block_size=1)

        self.mish = Activation(mish)

        self.conv2 = Conv2D(filters=filter, kernel_size=(3, 3), padding='valid', strides=(1, 1),
                            bias_initializer=None,kernel_initializer='he_uniform')
        self.bn2 = BatchNormalization()
        self.drop2 = DropBlock2D(keep_prob=0.8, block_size=3)
        self.mish2 = Activation(mish)

        self.G = GlobalAveragePooling2D()
        self.D = Dense(filter // 16, activation=mish)
        self.D2 = Dense(filter,activation='sigmoid')
        self.reshape =tf.keras.layers.Reshape((1,1,filter))
        self.max =MaxBlurPooling2D()
        self.conv3 = Conv2D(filters=filter, kernel_size=(1, 1), padding='same', strides=(1, 1),
                            bias_initializer=None,kernel_initializer='he_uniform')
        self.bn3 = BatchNormalization()
        self.drop3 = DropBlock2D(keep_prob=0.8, block_size=1)
        self.mish3 = Activation(mish)


    def call(self,inputs):
        a=[]
        x=self.conv1(inputs)
        x=self.bn(x)
        x = self.drop(x)
        x = self.mish(x)
        x = tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop2(x)
        x = self.mish2(x)

        a =self.G(x)
        a=self.D(a)
        a=self.D2(a)
        a=self.reshape(a)
        x = x*a
        x = self.max(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.drop3(x)
        x = self.mish3(x)

        return x
#



class Residual(tf.keras.Model):
    def __init__(self,filter):
        super(Residual,self).__init__()
        self.x =  AveragePooling2D(padding='same')
        self.x2 = Conv2D(filters=filter, kernel_size=(1, 1), padding='same', bias_initializer=None,)
        self.b = BatchNormalization()
        self.x3 = DropBlock2D(keep_prob=0.8, block_size=1)
        self.ac = Activation(mish)

    def call(self,inputs):

        residual = self.x(inputs)
        residual = self.x2(residual)
        residual = self.b(residual)
        residual = self.x3(residual)
        residual = self.ac(residual)

        return residual


class resnet(tf.keras.Model):
    def __init__(self,filter):
        super(resnet,self).__init__()
        self.x1 = Conv2D(filters=filter, kernel_size=(1, 1), padding='same',bias_initializer=None,)
        self.b1 = BatchNormalization()
        self.ac1 = Activation(mish)
        self.x2 = Conv2D(filters=filter//32, kernel_size=(3, 3), padding='same', bias_initializer=None, )
        self.b2 = BatchNormalization()
        self.ac2 = Activation(mish)
        self.G = GlobalAveragePooling2D()
        self.D = Dense(filter//32)
        self.b3 = BatchNormalization()
        self.ac3 = Activation(mish)
        self.reshape = tf.keras.layers.Reshape((1,1,filter))


    def call(self,inputs):
        next=[]
        d=[]
        residual = inputs
        x = self.x1(inputs)
        x = self.b1(x)
        x1 = self.ac1(x)
        for i in range(32):
            x = self.x2(x1)
            x = self.b2(x)
            x = self.ac2(x)
            next.append(x)
        next = tf.keras.layers.concatenate(next,axis=-1)
        x2 = self.G(next)
        for i in range(32):
            x = self.D(x2)
            x = self.b3(x)
            x = self.ac3(x)
            d.append(x)
        x=tf.keras.layers.concatenate(d,axis=-1)

        x =tf.keras.activations.softmax(x,axis=-1)

        x = self.reshape(x)

        x= next*x
        x = residual+x
        return x


class Assemble_resnet(tf.keras.Model):
    def __init__(self):
        super(Assemble_resnet,self).__init__()
        self.Stem = Stem()
        self.residual1 = Residual(128)
        self.ct1 = ct(128)

        self.res = resnet(128)
        # self.residual2 = Residual(384)
        # self.ct2 = ct(384)
        # self.res2 = resnet(384)
        # self.residual3 = Residual(768)
        # self.ct3 = ct(768)
        # self.res3 = resnet(768)
        self.G = GlobalAveragePooling2D()
        self.D = Dropout(0.2)
        self.fc = Dense(30, activation='softmax')

    def call(self,inputs):
        x = self.Stem(inputs)
        residual = self.residual1(x)
        x = self.ct1(x)
        x = tf.keras.layers.Add()([x, residual])
        for i in range(3):
            x = self.res(x)

        # residual = self.residual2(x)
        # x = self.ct2(x)
        # x = tf.keras.layers.Add()([x, residual])
        # for i in range(5):
        #     x = self.res2(x)
        #
        # residual = self.residual3(x)
        # x = self.ct3(x)
        # x = tf.keras.layers.Add()([x, residual])
        # for i in range(5):
        #     x = self.res3(x)

        x = self.G(x)
        x = self.D(x)
        x = self.fc(x)

        return x
#############################################################################
#Model------------------------------------------------------------------------------------------------------------------

with strategy.scope():
    i1 = Input(shape=input)
    x = Assemble_resnet()(i1)
    model = Model(inputs=i1, outputs=x)
    model.compile(RAdam(),loss=tf.keras.losses.kl_divergence)
    model.summary()


#Save-------------------------------------------------------------------------------------------------------------------
# model_name='sample1'
# model_path='./submission/model/'
# if not os.path.exists(model_path):
#     os.mkdir(model_path)
# submission_sample=pth.join(model_path,model_name)
# os.makedirs(submission_sample,exist_ok=True)
# checkpoint_path=pth.join(submission_sample,'Epoch_{epochs:03d}_Val_{val_loss:0.3f}.hdf5')
# checkpoint=ModelCheckpoint(checkpoint_path,verbose=1,save_best_only=True,monitor='val_loss')

# model_name = 'baseline_sampleCNN-original'
# model_path = 'submission/model/'
# if not os.path.exists(model_path):
#     os.mkdir(model_path)
# # Validation 점수가 가장 좋은 모델만 저장합니다.
# checkpoint_path = pth.join(model_path, model_name)
# os.makedirs(checkpoint_path, exist_ok=True)


def scheduler(epoch,lr):
    if epoch < 11:
        return lr

    else:
        return lr * tf.math.exp(-0.1)



callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
model_file_path = pth.join('D:\GAT\checkpo\check1', 'Epoch_{epoch:03d}_Val_{val_loss:.3f}.hdf5')
checkpoint = ModelCheckpoint(filepath=model_file_path, monitor='val_loss', verbose=1, save_weights_only=True)
#Train and Evaluate-----------------------------------------------------------------------------------------------------
history=model.fit(x_train,y_train,epochs=50,batch_size=batch_size,validation_data=(x_val,y_val),callbacks=[checkpoint])
scores=model.evaluate(xtest,ytest)
# history = model.fit_generator(train_Iterator,steps_per_epoch=len(train_Iterator),validation_data=valid_Iterator,validation_steps=len(valid_Iterator),epochs=30)
# scores=model.evaluate(x_test,y_test,batch_size=batch_size)
# model.save('D:\GAT\checkpo\check1\check2\model.hdf5')
# # #


from sklearn.metrics import mean_squared_error


#Prediction-------------------------------------------------------------------------------------------------------------
# submission_base=pth.join('submission')
# os.makedirs(submission_base,exist_ok=True)
# checkpoint_path=pth.join(model_path,model_name)
# weight_file=glob('{}/*hdf5'.format(checkpoint_path))[-1]
# model.load_weights(weight_file)
# y_pred=model.predict(predict_dataset)

# from tensorflow.keras.models import load_model
# model=load_model('D:/GAT/checkpo/check1/check2/Epoch_001_Val_1.477.hdf5',custom_objects={'mish' : mish, 'RAdam' : RAdam})
# y_pred=model.predict(xtest)
# # End--------------------------------------------------------------------------------------------------------------------
# submission=pd.read_csv('D:/GAT/subm/submission.csv', index_col=0)
# submission.loc[:,:]=y_pred
# submission.to_csv('D:/GAT/subm/subsubsubmissionall.csv')
# submission=submission[:10000]

# submission.loc[:,:]=y_pred
# submission.to_csv('D:/GAT/subm/subsubsubmission3.csv')


# submission.to_csv(pth.join(submission_base,'{}.csv'.format((model_name))))
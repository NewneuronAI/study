from keras.layers import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from keras.utils import plot_model
import tensorflow as tf
class StandardizedConv2D(tf.keras.layers.Conv2D):
  def build(self, input_shape):
    super(StandardizedConv2D, self).build(input_shape)
    default_conv_op = self._convolution_op
    def standardized_conv_op(inputs, kernel):
      mean, var = tf.nn.moments(kernel, axes=[0,1,2], keepdims=True)
      return default_conv_op(inputs, (kernel - mean) / tf.sqrt(var + 1e-10))

    self._convolution_op = standardized_conv_op
    self.built = True
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

# %%
class ResNet():
    """
    Usage:
        sr = ResNet([4,8,16], input_size=(50,50,1), output_size=12)
        sr.build()
        followed by sr.m.compile(loss='categorical_crossentropy',
                                 optimizer='adadelta', metrics=["accuracy"])
        save plotted model with:
            keras.utils.plot_model(sr.m, to_file = '<location>.png',
                                   show_shapes=True)
    """

    def __init__(self,
                 filters_list=[],
                 input_size=None,
                 output_size=None,
                 initializer='glorot_uniform'):
        self.filters_list = filters_list
        self.input_size = input_size
        self.output_size = output_size
        self.initializer = initializer
        self.m = None

    def _block(self, filters, inp):
        """ one residual block in a ResNet

        Args:
            filters (int): number of convolutional filters
            inp (tf.tensor): output from previous layer

        Returns:
            tf.tensor: output of residual block
        """
        layer_1 = BatchNormalization()(inp)
        act_1 = Activation('relu')(layer_1)
        conv_1 = Conv2D(filters, (3, 3),
                        padding='same',
                        kernel_initializer=self.initializer)(act_1)
        layer_2 = BatchNormalization()(conv_1)
        act_2 = Activation('relu')(layer_2)
        conv_2 = Conv2D(filters, (3, 3),
                        padding='same',
                        kernel_initializer=self.initializer)(act_2)
        return (conv_2)

    def build(self):
        """
        Returns:
            keras.engine.training.Model
        """
        i = Input(shape=self.input_size, name='input')
        x = Conv2D(self.filters_list[0], (3, 3),
                   padding='same',
                   kernel_initializer=self.initializer)(i)
        x = MaxPooling2D(padding='same')(x)
        x = Add()([self._block(self.filters_list[0], x), x])
        x = Add()([self._block(self.filters_list[0], x), x])
        x = Add()([self._block(self.filters_list[0], x), x])
        if len(self.filters_list) > 1:
            for filt in self.filters_list[1:]:
                x = Conv2D(filt, (3, 3),
                           strides=(2, 2),
                           padding='same',
                           activation='relu',
                           kernel_initializer=self.initializer)(x)
                x = Add()([self._block(filt, x), x])
                x = Add()([self._block(filt, x), x])
                x = Add()([self._block(filt, x), x])
        x = GlobalAveragePooling2D()(x)

        x = Dropout(0.5)(x)
        x = Dense(256,activation='relu')(x)
        x = Dense(self.output_size, activation='softmax')(x)

        self.m = Model(i, x)
        return self.m
        del self.m


# %%
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# %%
class CTC():
    """
    Usage:
        sr_ctc = CTC(enter input_size and output_size)
        sr_ctc.build()
        sr_ctc.m.compile()
        sr_ctc.tm.compile()
    """

    def __init__(self,
                 input_size=None,
                 output_size=None,
                 initializer='glorot_uniform'):
        self.input_size = input_size
        self.output_size = output_size
        self.initializer = initializer
        self.m = None
        self.tm = None

    def build(self,
              conv_filters=196,
              conv_size=13,
              conv_strides=4,
              act='relu',
              rnn_layers=2,
              LSTM_units=128,
              drop_out=0.8):
        """
        build CTC training model (self.m) and
        prediction model without the ctc loss function (self.tm)

        Usage:
            enter conv parameters for Cov1D layer
            specify number of rnn layers, LSTM units and dropout
        Args:

        Returns:
            self.m: keras.engine.training.Model
            self.tm: keras.engine.training.Model
        """
        i = Input(shape=self.input_size, name='input')
        x = Conv1D(conv_filters,
                   conv_size,
                   strides=conv_strides,
                   name='conv1d')(i)
        x = BatchNormalization()(x)
        x = Activation(act)(x)
        for _ in range(rnn_layers):
            x = Bidirectional(LSTM(LSTM_units,
                                   return_sequences=True))(x)
            x = Dropout(drop_out)(x)
            x = BatchNormalization()(x)
        y_pred = TimeDistributed(Dense(self.output_size,
                                       activation='softmax'))(x)
        # ctc inputs
        labels = Input(name='the_labels', shape=[None, ], dtype='int32')
        input_length = Input(name='input_length', shape=[1], dtype='int32')
        label_length = Input(name='label_length', shape=[1], dtype='int32')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(ctc_lambda_func,
                          output_shape=(1,),
                          name='ctc')([y_pred,
                                       labels,
                                       input_length,
                                       label_length])
        self.tm = Model(inputs=i,
                        outputs=y_pred)
        self.m = Model(inputs=[i,
                               labels,
                               input_length,
                               label_length],
                       outputs=loss_out)
        return self.m, self.tm

import os
import tensorflow as tf
import numpy as np
import random
import time
from struct import *
from pyswarm import pso

# Spare 15
def get_mask(layers, iterator):
    true_count = min(15, layers[iterator])
    false_count = max(0, layers[iterator] - true_count)
    return tf.Variable(
        [
            tf.random_shuffle(
                [True for _ in range(true_count)] + [False for _ in range(false_count)]
            ) for i in range(layers[iterator + 1])
        ]
    )


def selu(x, a=1.6733, l=1.0507):
    pos = l * tf.nn.relu(x)
    neg = l * (a * tf.exp((x - tf.abs(x)) * 0.5) - a)
    return pos + neg


def create_autoencoder_model(layers, x, activation, last_layer='linear'):
    size = (len(layers) - 1) * 2
    W = [None for _ in range(size)]
    b = [None for _ in range(size)]
    y = [None for _ in range(size)]
    mask = [None for _ in range(size)]
    for iterator in range(len(layers) - 1):

        if (activation == tf.nn.sigmoid):
            mask[iterator] = get_mask(layers, iterator)
            mask[size - iterator - 1] = get_mask(layers, iterator)

            W[iterator] = tf.Variable(
                tf.where(tf.transpose(mask[iterator]), tf.truncated_normal(
                    [layers[iterator], layers[iterator + 1]], 0, 1), tf.zeros([layers[iterator], layers[iterator + 1]]))
            )
        elif (activation == selu):
            W[iterator] = tf.Variable(
                tf.truncated_normal([layers[iterator], layers[iterator + 1]], 0, 1 / np.sqrt(layers[iterator]))
            )
            b[iterator] = tf.Variable(tf.zeros(layers[iterator + 1]))
        if iterator == 0:
            y[iterator] = activation(tf.matmul(x, W[iterator]) + b[iterator])
        elif iterator == len(layers) - 2:
            y[iterator] = tf.matmul(y[iterator - 1], W[iterator]) + b[iterator]
        else:
            y[iterator] = activation(tf.matmul(y[iterator - 1], W[iterator]) + b[iterator])

    for iterator in range(len(layers) - 1):
        if (activation == tf.nn.sigmoid):
            W[iterator + len(layers) - 1] = tf.Variable(
                tf.where(mask[iterator + len(layers) - 1], tf.truncated_normal(
                    [layers[len(layers) - iterator - 1], layers[len(layers) - iterator - 2]], 0, 1),
                         tf.zeros([layers[len(layers) - iterator - 1], layers[len(layers) - iterator - 2]]))
            )
        elif (activation == selu):
            W[iterator + len(layers) - 1] = tf.Variable(
                tf.truncated_normal(
                    [layers[len(layers) - iterator - 1], layers[len(layers) - iterator - 2]], 0,
                    1 / np.sqrt(layers[len(layers) - iterator - 1]))
            )
        b[iterator + len(layers) - 1] = tf.Variable(tf.zeros(layers[len(layers) - iterator - 2]))
        y[iterator + len(layers) - 1] = activation(
            tf.matmul(y[iterator + len(layers) - 2], W[iterator + len(layers) - 1]) + b[iterator + len(layers) - 1])

    if (last_layer == 'linear'):
        y[size - 1] = tf.matmul(y[size - 2], W[size - 1]) + b[size - 1]
    elif (last_layer == 'sigmoid'):
        y[size - 1] = tf.nn.sigmoid(tf.matmul(y[size - 2], W[size - 1]) + b[size - 1])
    elif (last_layer == 'selu'):
        y[size - 1] = selu(tf.matmul(y[size - 2], W[size - 1]) + b[size - 1])
    elif (last_layer == 'activation'):
        y[size - 1] = activation(tf.matmul(y[size - 2], W[size - 1]) + b[size - 1])
    else:
        y[size - 1] = activation(tf.matmul(y[size - 2], W[size - 1]) + b[size - 1])

    return W, b, y[-1]


class DataSet:
    input_size = 0
    output_size = 0
    samples_count = 0
    inputs = []
    outputs = []
    double_size = 8
    min_value = 0
    max_value = 0

    def __init__(self, file_path, zero_one):
        self.read_data_bin(file_path, zero_one)

    def scale_to_zero_one(self, data):
        data = (data - self.min_value) / (self.max_value - self.min_value)
        #         m = np.mean(data)
        #         sd = np.std(data)
        #         print("Mean = ", m)
        #         print("Std = ", sd)
        #         return (data - m)/sd
        return data

    #         return (data - self.min_value)/(self.max_value-self.min_value)

    def scale_to_original(self, data):
        return data * (self.max_value - self.min_value) + self.min_value

    def read_data_bin(self, file_path, zero_one):
        inputs = []
        outputs = []
        file = open(file_path, "rb")
        self.samples_count, self.input_size, self.output_size = unpack("iii", file.read(12))

        for i in range(self.samples_count):
            self.inputs.append(unpack(str(self.input_size) + "d", file.read(self.double_size * self.input_size)))
            self.outputs.append(unpack(str(self.output_size) + "d", file.read(self.double_size * self.output_size)))

        self.inputs = np.array(self.inputs)
        self.outputs = np.array(self.outputs)

        self.min_value = self.inputs.min()
        self.max_value = self.inputs.max()
        if (zero_one):
            self.inputs = self.scale_to_zero_one(self.inputs)

        file.close()

    def get_next_bach(self, count):
        indexes = random.sample(range(self.samples_count), count)
        return [self.inputs[indexes], self.outputs[indexes]]


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_set = DataSet("C:\Mgr\DataSets\Faces25x25.bin", False)
# data_set = DataSet("C:\Mgr\DataSets\HandwrittenDigitsMnist.bin", True)
# data_set = DataSet("C:\Mgr\DataSets\Curves.bin", True)

test = data_set.get_next_bach(10)[0]

# FACES 
input_dim = 625
autoencoder_layers = [625, 2000, 1000, 500, 30]
shape = (25, 25)
last_layer = 'linear'

# CURVES
# input_dim = 784
# autoencoder_layers = [784, 400, 200, 100, 50, 25, 5]
# shape = (28, 28)
# last_layer = 'sigmoid'

# MNIST
# input_dim = 784
# autoencoder_layers = [784, 1000, 500, 250, 30]
# shape = (28, 28)
# last_layer = 'sigmoid'


# MNIST
# input_dim = 784
# autoencoder_layers = [784, 500, 100, 2]
# shape = (28, 28)
# last_layer = 'sigmoid'

# activation = tf.nn.sigmoid
activation = selu
# activation = tf.nn.relu


x = tf.placeholder(tf.float32, [None, input_dim], name='x')

W, b, y = create_autoencoder_model(autoencoder_layers, x, activation, last_layer)

y_ = tf.placeholder(tf.float32, [None, input_dim])

error_avg_sqr = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y), axis=1))
error_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y) + (1 - y_) * tf.log(1 - y), axis=1))

moment = tf.placeholder(tf.float32, shape=[])
eps = tf.placeholder(tf.float32, shape=[])
train_step = tf.train.MomentumOptimizer(eps, moment, use_nesterov=False).minimize(error_avg_sqr)

time1 = time.time()
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
time2 = time.time()

print((time2 - time1))

batch_size = 200
print_info_rate = 10000
iteration_count = 15_000_000
max_moment = 0.99

eps_param = 0.00040

range_moment = 30_000

iterations = []
errors = []

for i in range(0, iteration_count + 1, batch_size):
    if i > 350_000:
        eps_param = 0.0001
    batch_xs, batch_ys = data_set.get_next_bach(batch_size)
    momentum = 1 - max(1 - max_moment, 0.5 / (i // range_moment + 1))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_xs, moment: momentum, eps: eps_param})
    if i % print_info_rate == 0:
        train_error_entropy = error_entropy.eval(feed_dict={
            x: batch_xs, y_: batch_xs})
        train_error_avg_sqr = error_avg_sqr.eval(feed_dict={
            x: batch_xs, y_: batch_xs})
        iterations.append(i)
        errors.append(train_error_avg_sqr)
        print('%d\t %.4f\t %.4f\t %.4f\t %f' % (i, train_error_entropy, train_error_avg_sqr, eps_param, momentum))

result = sess.run(y, feed_dict={x: test, y_: test})


# # Optymalizacja parametrów!
# # print(error(0.001, 0.99, 100_000, 0.5), error(0.0001, 0.99, 100_000, 0.5), error(0.01, 0.99, 100_000, 0.5))
#
#
# def error(params):
#     start_time = time.time()
#     eps_param = params[0]
#     range_moment = params[1]
#     max_moment = 0.99
#     start_moment = 0.5
#     batch_size = 200
#     iteration_count = 350_000
#     with tf.Session() as session:
#         tf.global_variables_initializer().run()
#         error = 0
#         for i in range(0, iteration_count+1, batch_size):
#             batch_xs, batch_ys = data_set.get_next_bach(batch_size)
#             momentum = 1 - max(1 - max_moment, start_moment / (i//range_moment + 1))
#             session.run(train_step, feed_dict={x: batch_xs, y_: batch_xs, moment: momentum, eps: eps_param})
#             error = error_avg_sqr.eval(feed_dict={x: batch_xs, y_: batch_xs})
# #             print(i)
# #             if i % 2 == 0:
#         end_time = time.time()
#         print("Błąd-", error, "-Eps-", eps_param, "-Zasieg-",range_moment,"-Max-",max_moment,"-Czas-", (end_time-start_time),)
#         return error
#
# lb = [0.000001, 25_000]
# ub = [0.000500, 100_000]
# xopt, fopt = pso(error, lb, ub, swarmsize = 10, maxiter = 100, debug = True)
# print(xopt, fopt)

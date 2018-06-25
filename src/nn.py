import os
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np
import random
import time
import functools
import math

from dataset import DataSet
from tqdm import tnrange, tqdm_notebook
from struct import *
from tensorflow.examples.tutorials.mnist import input_data
from pyswarm import pso
from tensorflow.python import debug as tf_debug

plt.rcParams['figure.figsize'] = (8, 10)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'



class NN: 
    
    norm = 0             
    optymalizer = 0      
    error_type = 0


    activation = None
    layers = [784, 1000, 500, 250, 30]
    input_dim = layers[0]
    shape = (int(np.sqrt(input_dim)), int(np.sqrt(input_dim)))
    last_layer_activation = 'sigmoid'
    
    batch_size = 200 
    iteration_count = 15_000_000    
    
#     g_param = 1
    eps_param = 0.005
    alfa_param = 0.00332143
    max_moment = 0.99
    range_moment = 50_000
    
    print_info_rate = 10_000      
    error_alpha = 0.2
    
    print_csv = True
    print_console = True
    print_charts = True
    
    data_set_file_name = "C:\Mgr\DataSets\HandwrittenDigitsMnist.bin"
    output_file_name = "output.csv"

    data_set= None;
    
    csv_separator = "------------"
    
    sess = None
    
    scale_to_zero_one = False
        
    def __init__(self, data_set_file_name, scale):
        self.data_set_file_name = data_set_file_name
        self.scale_to_zero_one = scale
        self.reload_data_set()

    def set_layers(self, layers_param):
        self.layers = layers_param
        self.input_dim = self.layers[0]
        self.shape = (int(np.sqrt(self.input_dim)), int(np.sqrt(self.input_dim)))
        
    def reload_data_set(self):
        self.data_set = DataSet(self.data_set_file_name, self.scale_to_zero_one)
        
    def selu(x, a = 1.6733, l = 1.0507):
        pos = l * tf.nn.relu(x)
        neg = l * (a * tf.exp( (x - tf.abs(x)) * 0.5 ) - a)
        return pos + neg

    def set_sigmoid_activation(self):
        self.activation = tf.nn.sigmoid

    def set_selu_activation(self):
        self.activation = NN.selu

    def generate_initial_value(self, iterator):
        true_count = min(15, self.layers[iterator])
        false_count = max(0, self.layers[iterator] - true_count)  

        value = [np.random.permutation(np.concatenate([np.random.normal(0, 1, true_count), [0]*false_count])).tolist()
                 for _ in range(self.layers[iterator + 1])]

        return  np.array(value).astype('float32')

    def show_images(self, images, cols_titles, x_size, y_size, name, dpi_i, title_size):
        cols = cols_titles

        if len(images.shape) == 2:
            columns = 1
            count = len(images)
        else:
            columns, count, _ = images.shape

        fig, axes = plt.subplots(nrows=count, ncols=columns, 
                                 figsize=(columns * y_size, count * x_size))

        for col in range(columns):
            for i in range(count):
                axes[i, col].axis('off')
                axes[i, col].imshow(np.reshape(images[col][i], self.shape), cmap='gray')
            
        for ax, col in zip(axes[0], cols):
            ax.set_title(col, fontsize=title_size)

        fig.tight_layout()
        
        plt.savefig(name, dpi=dpi_i)
        plt.show()
    
    
    def print_to_console(self, *text):
        if self.print_console:
            print(*text)
    
    def print_to_csv(self, *text):
        if self.print_csv:
            print(*text, 
                  sep=';',
                  file=open(self.output_file_name, "a"))
    
    def prepare_visualization(self):        
        if self.print_charts:
            self.fig, self.ax = plt.subplots(2,1)
            self.iterations = []
            self.errors = []
            self.mean_errors = []

        self.mean_train_error_avg_sqr = 0
        self.mean_train_error_entropy = 0

        self.print_information_about_nn()

    def print_information_about_nn(self):
        self.print_to_csv(
            self.csv_separator,
            self.layers,
            self.norm,
            self.optymalizer,
            self.activation.__name__,
#             self.g_param,
            self.eps_param,
            self.alfa_param,
            self.error_alpha,
            self.range_moment,
            self.last_layer_activation,
            self.error_type,
            self.batch_size,
            self.iteration_count,
            self.scale_to_zero_one
            )
    
    #Relikt - do posprzatania oraz potencjalna zmiana technologii!
    def plt_error(self, data_x, data_y):
        if self.ax[0].lines:
            for i in range(len(self.ax[0].lines)):
                self.ax[0].lines[i].set_xdata(data_x[-50:])
                self.ax[0].lines[i].set_ydata(data_y[i][-50:])
            self.ax[0].set_xlim(min(data_x[-50:]), max(data_x[-50:]))
            self.ax[0].set_ylim(0.9*np.min([data[-50:] for data in data_y]),
                           1.1*np.max([data[-50:] for data in data_y]))
        else:
            for line in data_y:
                self.ax[0].plot(data_x, line)
        if self.ax[1].lines:
            for i in range(len(self.ax[1].lines)):
                self.ax[1].lines[i].set_xdata(data_x[1:])
                self.ax[1].lines[i].set_ydata(data_y[i][1:])
            self.ax[1].set_xlim(data_x[1], data_x[-1])
            self.ax[1].set_ylim(0, np.max([data[1:] for data in data_y]))
        else:
            for line in data_y:
                self.ax[1].plot(data_x, line)
                
        self.fig.canvas.draw()

    def get_entropy_error(self, batch_xs):
        return self.error_entropy.eval(feed_dict={self.x: batch_xs, self.y_: batch_xs, self.iteration: self.i})#, self.g:self.g_param
    
    def get_error_avg_sqr(self, batch_xs):
        return self.error_avg_sqr.eval(feed_dict={self.x: batch_xs, self.y_: batch_xs, self.iteration: self.i})#, self.g:self.g_param
    
    def show_results(self, batch_xs, momentum, i):
    
        train_error_entropy = self.get_entropy_error(batch_xs)
        train_error_avg_sqr = self.get_error_avg_sqr(batch_xs)

        if i == self.print_info_rate or i == 0:
            self.mean_train_error_avg_sqr = train_error_avg_sqr
            self.mean_train_error_entropy = train_error_entropy
        else:
            self.mean_train_error_avg_sqr = (self.mean_train_error_avg_sqr * (1-self.error_alpha) + 
                                            train_error_avg_sqr * self.error_alpha)
            self.mean_train_error_entropy = (self.mean_train_error_entropy * (1-self.error_alpha) + 
                                            train_error_entropy * self.error_alpha)

            
        if self.print_charts:
            self.iterations.append(i)
            self.errors.append(train_error_avg_sqr)
            self.mean_errors.append(self.mean_train_error_avg_sqr)
            self.plt_error(self.iterations, [self.errors, self.mean_errors])
        
        self.print_to_console(
            '%d\t %.4f\t %.4f\t %.4f\t %.4f\t %f\t %f\t %d' % 
            (i, train_error_entropy, self.mean_train_error_entropy, train_error_avg_sqr, self.mean_train_error_avg_sqr, momentum, 1-tf.sigmoid((self.iteration-self.scale_w)*self.scale_alpha).eval(feed_dict={self.iteration: self.i}), self.i))
        self.print_to_csv(
            '%d;%.4f;%.4f;%.4f;%.4f;%f' % 
            (i, train_error_entropy, self.mean_train_error_entropy, train_error_avg_sqr, self.mean_train_error_avg_sqr, momentum))
            
        return self.mean_train_error_avg_sqr
        
    def run(self):
        
        start_time = time.time()

        if self.sess is not None:
            self.sess.close()
        
        tf.reset_default_graph()
        self.build_environment()
        
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        self.prepare_visualization()
        
        for self.i in tnrange(0, self.iteration_count+1, self.batch_size, desc='Learning'):
            batch_xs, batch_ys = self.data_set.get_next_bach(self.batch_size)
            
            self.momentum = 1 - max(1 - self.max_moment, 0.5 / (self.i//self.range_moment + 1))
            
            params = {
                self.x: batch_xs, 
                self.y_: batch_xs, 
                self.moment: 0, 
                self.eps: 1, 
#                 self.g:self.g_param, 
                self.iteration: self.i,#int(self.i/self.batch_size),
                self.alfa: self.alfa_param
            }
            
            if self.optymalizer == 0 or self.optymalizer == 1 or self.optymalizer == 2:
                params[self.moment] = self.momentum 
                params[self.eps] = self.eps_param
                
#             if i % self.print_info_rate == 0:
#                 self.show_results(batch_xs, self.momentum , self.i)
                
            self.sess.run([self.help_step], feed_dict = params)
            self.sess.run([self.train_step], feed_dict = params)

            if self.i % self.print_info_rate == 0:
                tmp = self.show_results(batch_xs, self.momentum, self.i)
                if math.isnan(tmp):
                    break  
        
        stop_time = time.time()
        self.print_to_console("Czas trwania:", stop_time - start_time)
        self.print_to_csv(self.csv_separator, stop_time - start_time)

    def continue_run(self):
        start_time = time.time()

        for self.i in tnrange(self.i, self.iteration_count+1, self.batch_size, desc='Learning'):
            batch_xs, batch_ys = self.data_set.get_next_bach(self.batch_size)
            
            self.momentum = 1 - max(1 - self.max_moment, 0.5 / (self.i//self.range_moment + 1))
            
            params = {
                self.x: batch_xs, 
                self.y_: batch_xs, 
                self.moment: 0, 
                self.eps: 1, 
#                 self.g:self.g_param, 
                self.iteration: self.i,#int(self.i/self.batch_size),
                self.alfa: self.alfa_param
            }
            
            if self.optymalizer == 0 or self.optymalizer == 1 or self.optymalizer == 2:
                params[self.moment] = self.momentum 
                params[self.eps] = self.eps_param
                
#             if i % self.print_info_rate == 0:
#                 self.show_results(batch_xs, self.momentum , self.i)
                
            self.sess.run([self.help_step], feed_dict = params)
            self.sess.run([self.train_step], feed_dict = params)

            if self.i % self.print_info_rate == 0:
                self.show_results(batch_xs, self.momentum, self.i)
        
        stop_time = time.time()
        self.print_to_console("Czas trwania:", stop_time - start_time)
        self.print_to_csv(self.csv_separator, stop_time - start_time)    
 
    def build_environment(self):
        self.x  = tf.placeholder(tf.float32, [None, self.input_dim], name = 'x')
        self.y_ = tf.placeholder(tf.float32, [None, self.input_dim], name = 'y_')
        #self.g  = tf.placeholder(tf.float32, name = 'g')

        self.alfa   = tf.placeholder(tf.float32, name = 'alfa')
        self.eps    = tf.placeholder(tf.float32, name = 'eps')
        self.moment = tf.placeholder(tf.float32, name = 'moment')

        self.iteration = tf.placeholder(tf.float32, shape=[], name = 'iteration')

        self.W, self.y, self.code_layer, self.layers_y = self.create_autoencoder_model()

        self.error_avg_sqr = tf.reduce_mean(
            tf.reduce_sum(tf.square(self.y_ - self.y), axis=1))

        self.error_entropy = tf.reduce_mean(
            -tf.reduce_sum(self.y_ * tf.log(self.y) + (1 - self.y_) * tf.log(1-self.y), axis=1))

        self.opimalizer = tf.train.MomentumOptimizer(self.eps, self.moment, use_nesterov = False)
        if self.error_type == 0:
            self.gradients = self.opimalizer.compute_gradients(self.error_entropy)
        if self.error_type == 1:
            self.gradients = self.opimalizer.compute_gradients(self.error_avg_sqr)
            
        self.help_step = []
        if self.optymalizer == 1: #Normalizacja
            self.gradients, self.help_step = self.normalize_gradients(self.gradients, 1)
        if self.optymalizer == 2: #Normalizacja
            self.gradients, self.help_step = self.normalize_gradients(self.gradients, 2)
        if self.optymalizer == 3: #ADAM
            self.gradients, self.help_step = self.adam_optimalizer(self.gradients)
        self.train_step = self.opimalizer.apply_gradients(self.gradients)
        
        if self.optymalizer == 4: #Adam TF
            if self.error_type == 0:
                self.train_step = tf.train.AdamOptimizer(self.alfa).minimize(self.error_entropy)
            if self.error_type == 1:
                self.train_step = tf.train.AdamOptimizer(self.alfa).minimize(self.error_avg_sqr)

    def create_autoencoder_model(self):  
        sigma_eps = 1e-15
        sigma = 1
        mi = 0

        size = (len(self.layers) - 1 ) * 2
        W = [None] * size
        y = [None] * size
        g = [None] * size

        for iterator in range(len(self.layers) - 1): 
            if (self.activation == tf.nn.sigmoid):
                W[iterator] = tf.Variable(
                    tf.concat([np.transpose(self.generate_initial_value(iterator)),
                               tf.zeros([1, self.layers[iterator + 1]])], 0),
                    name= "W" + str(iterator)
                )
            elif (self.activation == NN.selu):     
                W[iterator] = tf.Variable(
                    tf.concat([
                        tf.truncated_normal([self.layers[iterator],
                                             self.layers[iterator + 1]],
                                            0, 1/np.sqrt(self.layers[iterator])),
                        tf.zeros([1, self.layers[iterator + 1]])
                    ], 0),
                    name= "W" + str(iterator)
                )
            
            if self.norm == 1 or self.norm == 2 or self.norm == 3:
                g = tf.Variable(tf.ones(self.layers[iterator + 1]), name = "G" + str(iterator))
                self.scale_alpha = 1e-5
                self.scale_w = 1e6
                scale = 1-tf.sigmoid((self.iteration-self.scale_w)*self.scale_alpha)
            else:
                g = 1
                
            if iterator == 0:
                tmp = tf.matmul(self.x, W[iterator][0:-1])
                if self.norm == 1:
                    sigma = tf.norm(W[iterator][0:-1] + sigma_eps, axis = 0, name = "test2")
                    g_sigma = g/sigma
                if self.norm == 2:
                    mi = tf.transpose(
                        tf.tile([tf.reduce_mean(tmp, 1)], [tf.shape(W[iterator])[1], 1]))
                    sigma = tf.transpose(
                        tf.tile([tf.sqrt(tf.reduce_mean(tf.square(tmp - mi), 1))], [tf.shape(W[iterator])[1], 1]))
                    g_sigma = g/sigma
                if self.norm == 3:
                    mi = tf.tile([tf.reduce_mean(tmp, 0)], [tf.shape(self.x)[0], 1])
                    sigma = tf.tile([tf.sqrt(keras.backend.var(tmp, 0) + sigma_eps)], [tf.shape(self.x)[0], 1])
                    g_sigma = tf.sign(g)*(tf.pow(tf.abs(g)/sigma, scale))
                    mi = mi * scale
                    
                y[iterator] = self.activation((g_sigma) * (tmp - mi) + W[iterator][-1])

            elif iterator == len(self.layers) - 2:
                tmp = tf.matmul(y[iterator-1], W[iterator][0:-1]) 
                if self.norm == 1:
                    sigma = tf.norm(W[iterator][0:-1]+ sigma_eps, axis = 0)
                    g_sigma = g/sigma
                if self.norm == 2:
                    mi = tf.transpose(tf.tile([tf.reduce_mean(tmp, 1)], [tf.shape(W[iterator])[1], 1]))
                    sigma = tf.transpose(
                        tf.tile([tf.sqrt(tf.reduce_mean(tf.square(tmp - mi), 1))], [tf.shape(W[iterator])[1], 1]))
                    g_sigma = g/sigma
                if self.norm == 3:
                    mi = tf.tile([tf.reduce_mean(tmp, 0)], [tf.shape(y[iterator-1])[0], 1])
                    sigma = tf.tile([tf.sqrt(keras.backend.var(tmp, 0) + sigma_eps)], [tf.shape(y[iterator-1])[0], 1]) 
                    g_sigma = tf.sign(g)*(tf.pow(tf.abs(g)/sigma, scale))
                    mi = mi * scale

                y[iterator] = (g_sigma) * (tf.matmul(y[iterator-1], W[iterator][0:-1]) - mi) + W[iterator][-1]

            else:
                tmp = tf.matmul(y[iterator-1], W[iterator][0:-1])           
                if self.norm == 1:
                    sigma = tf.norm(W[iterator][0:-1] + sigma_eps, axis = 0) 
                    g_sigma = g/sigma
                if self.norm == 2:
                    mi = tf.transpose(
                        tf.tile([tf.reduce_mean(tmp, 1)], [tf.shape(W[iterator])[1], 1]))
                    sigma = tf.transpose(
                        tf.tile([tf.sqrt(tf.reduce_mean(tf.square(tmp - mi), 1))], [tf.shape(W[iterator])[1], 1]))
                    g_sigma = g/sigma
                if self.norm == 3:
                    mi = tf.tile([tf.reduce_mean(tmp, 0)], [tf.shape(y[iterator-1])[0], 1])
                    sigma = tf.tile([tf.sqrt(keras.backend.var(tmp, 0) + sigma_eps)], [tf.shape(y[iterator-1])[0], 1])
                    g_sigma = tf.sign(g)*(tf.pow(tf.abs(g)/sigma, scale))
                    mi = mi * scale

                y[iterator] = self.activation((g_sigma) * (tmp - mi) + W[iterator][-1])

        for iterator in range(len(self.layers) - 1):
            if (self.activation == tf.nn.sigmoid):
                W[iterator + len(self.layers) - 1] = tf.Variable(
                    tf.concat([
                        self.generate_initial_value(len(self.layers) - 2 - iterator),
                        tf.zeros([1, self.layers[len(self.layers) - iterator - 2]])
                    ], 0),    
                    name= "W" + str(iterator + len(self.layers) - 1)
                )
            elif (self.activation == NN.selu):
                W[iterator + len(self.layers) - 1] = tf.Variable(
                    tf.concat([
                        tf.truncated_normal(
                            [self.layers[len(self.layers) - iterator - 1],
                             self.layers[len(self.layers) - iterator - 2]],
                            0, 
                            1/np.sqrt(self.layers[len(self.layers) - iterator - 1])),
                        tf.zeros([1, self.layers[len(self.layers) - iterator - 2]])
                    ], 0),
                    name= "W" + str(iterator + len(self.layers) - 1)
                )

            if self.norm == 1 or self.norm == 2 or self.norm == 3:
                g = tf.Variable(tf.ones(self.layers[len(self.layers) - iterator - 2]), name = "G" + str(iterator + len(self.layers) - 1))
            else:
                g = 1
                
            tmp = tf.matmul(y[iterator + len(self.layers) - 2], W[iterator + len(self.layers) - 1][0:-1]) 
            if self.norm == 1:
                sigma = tf.norm(W[iterator + len(self.layers) - 1][0:-1] + sigma_eps, axis = 0) 
                g_sigma = g/sigma
            if self.norm == 2:
                mi = tf.transpose(
                    tf.tile([tf.reduce_mean(tmp, 1)], [tf.shape(W[iterator + len(self.layers) - 1])[1], 1]))
                sigma = tf.transpose(
                    tf.tile(
                        [tf.sqrt(tf.reduce_mean(tf.square(tmp - mi), 1))],
                        [tf.shape(W[iterator + len(self.layers) - 1])[1], 1]
                    )
                )
                g_sigma = g/sigma
            if self.norm == 3:
                mi = tf.tile([tf.reduce_mean(tmp, 0)], [tf.shape(y[iterator + len(self.layers) - 2])[0], 1])
                sigma = tf.tile([tf.sqrt(keras.backend.var(tmp, 0) + sigma_eps)], 
                                [tf.shape(y[iterator + len(self.layers) - 2])[0], 1])       
                g_sigma = tf.sign(g)*(tf.pow(tf.abs(g)/sigma, scale))
                mi = mi * scale
                
            y[iterator + len(self.layers) - 1] = self.activation(
                (g_sigma) * (tmp - mi) + W[iterator + len(self.layers) - 1][-1]
            )

        if(self.last_layer_activation == 'linear'):
            y[size - 1] = tf.matmul(y[size - 2], W[size - 1][0:-1]) + W[size - 1][-1]

        elif(self.last_layer_activation == 'sigmoid'):
            y[size - 1] = tf.nn.sigmoid(
                tf.matmul(y[size - 2], W[size - 1][0:-1]) + W[size - 1][-1]
            )

        elif(self.last_layer_activation == 'selu'):
            y[size - 1] = selu(
                tf.matmul(y[size - 2], W[size - 1][0:-1]) + W[size - 1][-1]
            )

        else:
            y[size - 1] = self.activation(
                tf.matmul(y[size - 2], W[size - 1][0:-1]) + W[size - 1][-1]
            )

        return W, y[-1], y[len(self.layers) - 2], y        
         
    def adam_optimalizer(self, gradients):
        b1 = 0.9
        b2 = 0.999
        e = 1e-8
        result = []
        to_do = []
        counter = tf.Variable(0.0)
        for i in range(len(gradients)):
            if gradients[i][0] != None:
                m = tf.Variable(tf.zeros(gradients[i][0].shape), name = "m"+str(i))
                v = tf.Variable(tf.zeros(gradients[i][0].shape), name = "v"+str(i))

                update_m = tf.assign(m, b1 * m + (1 - b1) * gradients[i][0])
                update_v = tf.assign(v, b2 * v + (1 - b2) * tf.square(gradients[i][0]))
                
                to_do.append(update_m)
                to_do.append(update_v)

                alfa_t = self.alfa * tf.sqrt(1-tf.pow(b2, counter))/(1 - tf.pow(b1, counter))

                result.append((tf.divide(m * alfa_t, tf.sqrt(v) + e), gradients[i][1]))
                
        to_do.append(tf.assign(counter, counter+1))
        return result, to_do   
    
    def normalize_gradients(self, gradients, function):
        e = 1e-8
        grad_count = (len(self.layers)-1)*2
        self.grad_norms = [tf.Variable(1.0) for _ in range(grad_count)]
        increment_grad_norms = []

        if function == 1:
            for i in range(len(self.grad_norms) - 1):
                if i != len(self.layers) - 2:
                    increment_grad_norms.append(
                        tf.assign(self.grad_norms[i], (self.grad_norms[i] * (1 - self.alfa)) +
                                  (keras.backend.std(self.layers_y[i])) * self.alfa)
                    )

        if function == 2:
            for i in range(len(self.grad_norms) - 1):
                if i != len(self.layers) - 2:
                    increment_grad_norms.append(
                        tf.assign(self.grad_norms[i], 
                                  (self.grad_norms[i] * (1 - self.alfa)) + 
                                  tf.square(tf.reduce_mean(tf.square(gradients[i][0][0:-1]))) * self.alfa)
                    )

        result = []

        for i in range(len(gradients)):
            if "W" in gradients[i][1].name:
                index = int(gradients[i][1].name[1])
                result.append((tf.concat([tf.divide(gradients[i][0][0:-1], self.grad_norms[index]), gradients[i][0][-2:-1]], axis = 0),
                               gradients[i][1]))
            else:
                result.append(gradients[i])
        return result, increment_grad_norms
    
    def optymalize_paramteres(self, parameters, lb, ub, swarmsize_param, maxiter_param, debug_param):
        old_params = [self.print_csv, self.print_console, self.print_charts]
        self.print_information_about_nn()
        self.print_csv = False
        self.print_console = False
        self.print_charts = False
        
        xopt, fopt = pso(functools.partial(self.error_function, parameters), lb, ub, 
                         swarmsize = swarmsize_param, 
                         maxiter = maxiter_param, 
                         debug = debug_param)
        
        self.print_csv, self.print_console, self.print_charts = old_params
        
        self.print_to_console("Parametry:", xopt, "Wynik", fopt)
        self.print_to_csv(parameters, xopt, fopt, swarmsize_param, maxiter_param)
    
    # which_params - eps, moment, alfa, range_moment   
    def error_function(self, which_params, params):
        old_params = [self.eps_param, self.alfa_param, self.range_moment, self.moment_param]
        i = 0
        if which_params[0]:
            self.eps_param = params[i]
            i = i + 1
        if which_params[1]:
            self.moment_param = params[i]
            i = i + 1
        if which_params[2]:
            self.alfa_param = params[i]
            i = i + 1
        if which_params[3]:
            self.range_moment = params[i]
            i = i + 1

        self.run()
        
        self.eps_param, self.alfa_param, self.range_moment, self.moment_param = old_params
        
        batch_xs, _ = self.data_set.get_next_bach(self.batch_size)
        
        return self.mean_train_error_avg_sqr
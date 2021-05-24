

from load_cora import normalize_adj_numpy


def graph_conv_op(x, num_filters, graph_conv_filters, kernel):

    if len(x.get_shape()) == 2:
        conv_op = K.dot(graph_conv_filters, x)
        conv_op = tf.split(conv_op, num_filters, axis=0)
        conv_op = K.concatenate(conv_op, axis=1)
    elif len(x.get_shape()) == 3:
        conv_op = K.batch_dot(graph_conv_filters, x)
        conv_op = tf.split(conv_op, num_filters, axis=1)
        conv_op = K.concatenate(conv_op, axis=2)
    else:
        raise ValueError('x must be either 2 or 3 dimension tensor'
                         'Got input shape: ' + str(x.get_shape()))

    conv_out = K.dot(conv_op, kernel)
    return conv_out

from keras import activations, initializers, constraints, optimizers
from keras import regularizers
import keras.backend as K
from keras.engine.topology import Layer
import tensorflow as tf

class MultiGraphCNN(Layer):

    def __init__(self,
                 output_dim,
                 num_filters,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MultiGraphCNN, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.num_filters = num_filters

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_initializer.__name__ = kernel_initializer
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):

        if self.num_filters != int(input_shape[1][-2]/input_shape[1][-1]):
            raise ValueError('num_filters does not match with graph_conv_filters dimensions.')

        self.input_dim = input_shape[0][-1]
        kernel_shape = (self.num_filters * self.input_dim, self.output_dim)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs):

        output = graph_conv_op(inputs[0], self.num_filters, inputs[1], self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0], input_shape[0][1], self.output_dim)
        return output_shape

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'num_filters': self.num_filters,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(MultiGraphCNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


from tensorflow import keras
from keras.models import Input, Model, Sequential
from keras.layers import Dense, Activation, Dropout, Lambda, LSTM, SimpleRNN, GRU, LeakyReLU
import keras.backend as K
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import math

from utils import *

# prepare data
def preprocess_adj_tensor_with_identity(adj_tensor, symmetric=True):
    adj_out_tensor = []
    for i in range(adj_tensor.shape[0]):
        adj = adj_tensor[i]
        adj = adj + np.eye(adj.shape[0])
        adj = normalize_adj_numpy(adj, symmetric)
        adj = np.concatenate([np.eye(adj.shape[0]), adj], axis=0)
        adj_out_tensor.append(adj)
    adj_out_tensor = np.array(adj_out_tensor)
    return adj_out_tensor


# A = pd.read_csv('/media/ysh1/Seagate Backup Plus Drive/processed_taxi/6. Spatial dependency modeling/input_conn.csv')
A = pd.read_csv('/home/ysh1/PycharmProjects/thesis/time_input_norm_conn.csv', header=None)
A = np.array(A)
A = A[0:215100,]
num_graph_nodes = A.shape[1]
num_graphs = int(A.shape[0]/A.shape[1])

A = np.split(A, num_graphs, axis=0)
A = np.array(A)

A1 = pd.read_csv('/home/ysh1/PycharmProjects/thesis/adj_input_nei.csv')
A1 = np.array(A1)
A1 = A1[0:215100,]
num_graph_nodes = A1.shape[1]
num_graphs = int(A1.shape[0]/A1.shape[1])

A1 = np.split(A1, num_graphs, axis=0)
A1 = np.array(A1)

X = pd.read_csv('/home/ysh1/PycharmProjects/thesis/input.csv', header=None)
X = np.array(X)
X = np.split(X, num_graphs, axis=0)
X = np.array(X)

Y = pd.read_csv('/home/ysh1/PycharmProjects/thesis/input_label.csv', header=None)
Y = np.array(Y)
Y = Y[:,1]
Y = Y.reshape(717,300)

# A, X, Y = shuffle(A, X, Y)
# ratio = [0.5, 0.6, 0.7, 0.8, 0.9, 0.875]

Result = []
Result = np.array(Result)
# for i in range(len(ratio)):

train_size = 626
train_A = A[0:train_size]
test_A = A[train_size:len(A)]

train_A1 = A1[0:train_size]
test_A1 = A1[train_size:len(A1)]

train_X = X[0:train_size]
test_X = X[train_size:len(X)]

train_Y = Y[0:train_size]
test_Y = Y[train_size:len(Y)]

# date = [91,83,92,91,91,86,92,91]
# if test_month == 1:
# test_A = A[0:date[0]]
# train_A = A[sum(date[0:1]):len(A)]
#
# test_A1 = A1[0:date[0]]
# train_A1 = A1[sum(date[0:1]):len(A1)]
#
# test_X = X[0:date[0]]
# train_X = X[sum(date[0:1]):len(X)]
#
# test_Y = Y[0:date[0]]
# train_Y = Y[sum(date[0:1]):len(Y)]
#
# elif test_month == len(date):
# test_A = A[sum(date[0:(len(date)-1)]):len(A)]
# train_A = A[0:sum(date[0:(len(date)-1)])]
#
# test_A1 = A1[sum(date[0:(len(date)-1)]):len(A1)]
# train_A1 = A1[0:sum(date[0:(len(date)-1)])]
#
# test_X = X[sum(date[0:(len(date)-1)]):len(X)]
# train_X = X[0:sum(date[0:(len(date)-1)])]
#
# test_Y = Y[sum(date[0:(len(date)-1)]):len(Y)]
# train_Y = Y[0:sum(date[0:(len(date)-1)])]
#
# else:
# test_A = A[sum(date[0:(test_month-1)]):sum(date[0:(test_month)])]
# train_A = np.append(A[0:sum(date[0:(test_month-1)])], A[sum(date[0:(test_month)]):len(A)])
# train_A = train_A.reshape(len(A)-date[test_month-1],len(A[1,:]),len(A[1,:]))
#
# test_A1 = A1[sum(date[0:(test_month-1)]):sum(date[0:(test_month)])]
# train_A1 = np.append(A1[0:sum(date[0:(test_month-1)])], A1[sum(date[0:(test_month)]):len(A1)])
# train_A1 = train_A1.reshape(len(A1)-date[test_month-1],len(A1[1,:]),len(A1[1,:]))
#
# test_X = X[sum(date[0:(test_month-1)]):sum(date[0:(test_month)])]
# train_X = np.append(X[0:sum(date[0:(test_month-1)])], X[sum(date[0:(test_month)]):len(X)])
# train_X = train_X.reshape(len(X)-date[test_month-1],len(X[1,:]),len(X[1,:]))
#
# test_Y = Y[sum(date[0:(test_month-1)]):sum(date[0:(test_month)])]
# train_Y = np.append(Y[0:sum(date[0:(test_month-1)])], Y[sum(date[0:(test_month)]):len(Y)])
# train_Y = train_Y.reshape(len(Y)-date[test_month-1],len(Y[1,:]))

#######################GCN-RNN#########################


SYM_NORM = True
num_filters = 2
graph_conv_filters = preprocess_adj_tensor_with_identity(train_A, SYM_NORM)

# build model
X_input = Input(shape=(train_X.shape[1], train_X.shape[2]))
graph_conv_filters_input = Input(shape=(graph_conv_filters.shape[1], graph_conv_filters.shape[2]))
# graph_conv_filters_input = Input(shape=(train_A.shape[1], train_A.shape[2]))

nb_epochs = 1000
# hidden_size = 100
# dim = 300
batch_size = 1 # round(0.05*len(train_A))

output = MultiGraphCNN(20, num_filters, activation='relu')([X_input, graph_conv_filters_input])
output = Dropout(0.2)(output)
output = MultiGraphCNN(20, num_filters, activation='relu')([output, graph_conv_filters_input])
output = Dropout(0.2)(output)
output = Lambda(lambda x: K.mean(x, axis=1))(output)  # adding a node invariant layer to make sure output does not depends upon the node order in a graph.
output = Dense(train_Y.shape[1])(output)
output = LeakyReLU(0.2)(output)

#
# output = np.array(output)
# output = output.reshape(None,1,300)
# output = tf.convert_to_tensor(output)
#
# output = LSTM(hidden_size, input_shape=(1, dim), activation='hard_sigmoid')(output)
# output = Dense(dim)(output)

model = Model(inputs=[X_input, graph_conv_filters_input], outputs=output)

# RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
# sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
# Adagrad = optimizers.Adagrad(learning_rate=0.01)
adam = optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
# model.summary()

model.fit([train_X, graph_conv_filters], train_Y, batch_size=batch_size, epochs=nb_epochs, verbose=1)
# model.fit([train_X, train_A], train_Y, batch_size=batch_size, epochs=nb_epochs, verbose=1)

graph_conv_filters_test = preprocess_adj_tensor_with_identity(test_A, SYM_NORM)
testPredict = model.predict([test_X, graph_conv_filters_test])
trainPredict = model.predict([train_X, graph_conv_filters])

# trainPredict = model.predict([train_X, train_A])
# testPredict = model.predict([test_X, test_A])

###############################################
lr = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
epoch = [50,100,200,400,800,1000,1600,3200,5000]
layer = [1,2,4,5,8,10,12,15,20,40]
drop = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]


Result = []
Result = np.array(Result)


K.clear_session()

SYM_NORM = True
num_filters = 2
graph_conv_filters = preprocess_adj_tensor_with_identity(train_A1, SYM_NORM)

# build model
X_input = Input(shape=(train_X.shape[1], train_X.shape[2]))
graph_conv_filters_input = Input(shape=(graph_conv_filters.shape[1], graph_conv_filters.shape[2]))
# graph_conv_filters_input = Input(shape=(train_A1.shape[1], train_A1.shape[2]))

nb_epochs = 3200
# hidden_size = 100
# dim = 300
batch_size = 1 # round(0.05*len(train_A))

output = MultiGraphCNN(20, num_filters, activation='relu')([X_input, graph_conv_filters_input])
output = Dropout(0.2)(output)
output = MultiGraphCNN(20, num_filters, activation='relu')([output, graph_conv_filters_input])
output = Dropout(0.2)(output)
output = Lambda(lambda x: K.mean(x, axis=1))(output)  # adding a node invariant layer to make sure output does not depends upon the node order in a graph.
output = Dense(train_Y.shape[1])(output)
output = LeakyReLU(0.2)(output)

# output = np.array(output)
# output = output.reshape(None,1,300)
# output = tf.convert_to_tensor(output)
#
# output = LSTM(hidden_size, input_shape=(1, dim), activation='hard_sigmoid')(output)
# output = Dense(dim)(output)

model = Model(inputs=[X_input, graph_conv_filters_input], outputs=output)

# RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
# sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
# Adagrad = optimizers.Adagrad(learning_rate=0.01)
adam = optimizers.Adam(learning_rate=0.2, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
# model.summary()

model.fit([train_X, graph_conv_filters], train_Y, batch_size=batch_size, epochs=nb_epochs, verbose=1)
# model.fit([train_X, train_A1], train_Y, batch_size=batch_size, epochs=nb_epochs, verbose=1)

graph_conv_filters_test = preprocess_adj_tensor_with_identity(test_A1, SYM_NORM)
testPredict1 = model.predict([test_X, graph_conv_filters_test])
trainPredict1 = model.predict([train_X, graph_conv_filters])

# trainPredict1 = model.predict([train_X, train_A1])
# testPredict1 = model.predict([test_X, test_A1])


# train parameters
timestep = seq = num_graphs
dim = 2
train_size = trainPredict.shape[0]
test_size = testPredict.shape[0]

look_back = area = num_graph_nodes


temp = trainPredict.reshape(train_size*area,1)
temp1 = trainPredict1.reshape(train_size*area,1)
temp2 = train_Y.reshape(train_size*area,1)
temp3 = pd.concat([pd.DataFrame(temp), pd.DataFrame(temp1), pd.DataFrame(temp2)], axis=1, ignore_index=False)
train = np.array(temp3)

temp = testPredict.reshape(test_size*area,1)
temp1 = testPredict1.reshape(test_size*area,1)
temp2 = test_Y.reshape(test_size*area,1)
temp3 = pd.concat([pd.DataFrame(temp), pd.DataFrame(temp1), pd.DataFrame(temp2)], axis=1, ignore_index=False)
test = np.array(temp3)


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(int(len(dataset)/look_back)):
        a = dataset[look_back*i:look_back*(i+1),0:dim]
        dataX.append(a)
        dataY.append(dataset[look_back*i:look_back*(i+1), dim])
    return np.array(dataX), np.array(dataY)

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape for LSTM  -> (size, timestep, dim) = (150, year, 99)
trainX_t, trainY_t = [],[]
testX_t, testY_t = [],[]

for k in range(train_size):
    temp_t = trainX[k].reshape(area, 1, dim)
    trainX_t.append(temp_t)
    target_t = trainY[k].reshape(area, 1)
    trainY_t.append(target_t)

for k in range(test_size):
    temp_t = testX[k].reshape(area, 1, dim)
    testX_t.append(temp_t)
    target_t = testY[k].reshape(area, 1)
    testY_t.append(target_t)


# build LSTM from keras
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import GRU
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import keras.backend as K
import time

lr = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
epoch = [10, 20, 30, 50,80,100, 150, 200, 400]
batch = [1,2,5,10,20,30,40,50, 75, 100, 150, 200]
hidden_size = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 150]
Result = []
Result = np.array(Result)

K.clear_session()
model = Sequential()  # Sequeatial Model
model.add(LSTM(150, input_shape=(1, dim), activation='hard_sigmoid'))  # (timestep, feature)
# model.add(GRU(150, input_shape=(1,dim), activation='hard_sigmoid')) # (timestep, feature)
model.add(Dense(1))  # output = 31
RMSprop = optimizers.RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='mean_absolute_percentage_error', optimizer='RMSprop', metrics=['accuracy'])
model.summary()

start = time.time()
for k in range(train_size):
    print(k)
    model.fit(trainX_t[k], trainY_t[k], epochs=30, verbose=2, batch_size=150)

# training
# trainPredict = model.predict(trainX_t)
# testPredict_rnn = model.predict(testX_t)

testPredict_rnn = []
testPredict_rnn = np.array(testPredict_rnn)
for k in range(test_size):
    print(k)
    temp = model.predict(testX_t[k])
    testPredict_rnn = np.append(testPredict_rnn, temp)
testPredict_rnn = testPredict_rnn.reshape(test_size, area)

end = time.time()

real_pred = []
real_pred = np.array(real_pred)
testY_t = np.array(testY_t).reshape(test_size, area)
for k in range(test_size):
    print(k)
    temp = testY_t[k] - testPredict_rnn[k]
    real_pred = np.append(real_pred, temp)
real_pred = real_pred.reshape(test_size, area)

RMSE = []
RMSE = np.array(RMSE)
for k in range(area):
    print(k)
    temp = (sum(real_pred[:, k] ** 2) / test_size) ** 0.5
    RMSE = np.append(RMSE, temp)

MAPE = []
MAPE = np.array(MAPE)
for k in range(area):
    print(k)
    temp = 100 * sum(abs(real_pred[:, k]) / testY_t[:, k]) / test_size
    MAPE = np.append(MAPE, temp)

temp = [np.mean(RMSE), np.ma.masked_invalid(MAPE).mean(), np.mean(MAPE),  (end - start) / 60]
print(temp)

Result = np.append(Result, temp)
Result = Result.reshape(int(len(Result) / len(temp)), len(temp))




for i in range(len(hidden_size)):
    for j in range(len(batch)):SYM_NORM = True
num_filters = 2
graph_conv_filters = preprocess_adj_tensor_with_identity(train_A, SYM_NORM)

# build model
X_input = Input(shape=(train_X.shape[1], train_X.shape[2]))
graph_conv_filters_input = Input(shape=(graph_conv_filters.shape[1], graph_conv_filters.shape[2]))
# graph_conv_filters_input = Input(shape=(train_A.shape[1], train_A.shape[2]))

nb_epochs = 1000
# hidden_size = 100
# dim = 300
batch_size = 1 # round(0.05*len(train_A))

output = MultiGraphCNN(20, num_filters, activation='relu')([X_input, graph_conv_filters_input])
output = Dropout(0.2)(output)
output = MultiGraphCNN(20, num_filters, activation='relu')([output, graph_conv_filters_input])
output = Dropout(0.2)(output)
output = Lambda(lambda x: K.mean(x, axis=1))(output)  # adding a node invariant layer to make sure output does not depends upon the node order in a graph.
output = Dense(train_Y.shape[1])(output)
output = LeakyReLU(0.2)(output)

#
# output = np.array(output)
# output = output.reshape(None,1,300)
# output = tf.convert_to_tensor(output)
#
# output = LSTM(hidden_size, input_shape=(1, dim), activation='hard_sigmoid')(output)
# output = Dense(dim)(output)

model = Model(inputs=[X_input, graph_conv_filters_input], outputs=output)

# RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
# sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
# Adagrad = optimizers.Adagrad(learning_rate=0.01)
adam = optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
# model.summary()

model.fit([train_X, graph_conv_filters], train_Y, batch_size=batch_size, epochs=nb_epochs, verbose=1)
# model.fit([train_X, train_A], train_Y, batch_size=batch_size, epochs=nb_epochs, verbose=1)

graph_conv_filters_test = preprocess_adj_tensor_with_identity(test_A, SYM_NORM)
testPredict = model.predict([test_X, graph_conv_filters_test])
trainPredict = model.predict([train_X, graph_conv_filters])

# trainPredict = model.predict([train_X, train_A])
# testPredict = model.predict([test_X, test_A])

###############################################
lr = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
epoch = [50,100,200,400,800,1000,1600,3200,5000]
layer = [1,2,4,5,8,10,12,15,20,40]
drop = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]


Result = []
Result = np.array(Result)


K.clear_session()

SYM_NORM = True
num_filters = 2
graph_conv_filters = preprocess_adj_tensor_with_identity(train_A1, SYM_NORM)

# build model
X_input = Input(shape=(train_X.shape[1], train_X.shape[2]))
graph_conv_filters_input = Input(shape=(graph_conv_filters.shape[1], graph_conv_filters.shape[2]))
# graph_conv_filters_input = Input(shape=(train_A1.shape[1], train_A1.shape[2]))

nb_epochs = 3200
# hidden_size = 100
# dim = 300
batch_size = 1 # round(0.05*len(train_A))

output = MultiGraphCNN(20, num_filters, activation='relu')([X_input, graph_conv_filters_input])
output = Dropout(0.2)(output)
output = MultiGraphCNN(20, num_filters, activation='relu')([output, graph_conv_filters_input])
output = Dropout(0.2)(output)
output = Lambda(lambda x: K.mean(x, axis=1))(output)  # adding a node invariant layer to make sure output does not depends upon the node order in a graph.
output = Dense(train_Y.shape[1])(output)
output = LeakyReLU(0.2)(output)

# output = np.array(output)
# output = output.reshape(None,1,300)
# output = tf.convert_to_tensor(output)
#
# output = LSTM(hidden_size, input_shape=(1, dim), activation='hard_sigmoid')(output)
# output = Dense(dim)(output)

model = Model(inputs=[X_input, graph_conv_filters_input], outputs=output)

# RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
# sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
# Adagrad = optimizers.Adagrad(learning_rate=0.01)
adam = optimizers.Adam(learning_rate=0.2, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
# model.summary()

model.fit([train_X, graph_conv_filters], train_Y, batch_size=batch_size, epochs=nb_epochs, verbose=1)
# model.fit([train_X, train_A1], train_Y, batch_size=batch_size, epochs=nb_epochs, verbose=1)

graph_conv_filters_test = preprocess_adj_tensor_with_identity(test_A1, SYM_NORM)
testPredict1 = model.predict([test_X, graph_conv_filters_test])
trainPredict1 = model.predict([train_X, graph_conv_filters])

# trainPredict1 = model.predict([train_X, train_A1])
# testPredict1 = model.predict([test_X, test_A1])


# train parameters
timestep = seq = num_graphs
dim = 2
train_size = trainPredict.shape[0]
test_size = testPredict.shape[0]

look_back = area = num_graph_nodes


temp = trainPredict.reshape(train_size*area,1)
temp1 = trainPredict1.reshape(train_size*area,1)
temp2 = train_Y.reshape(train_size*area,1)
temp3 = pd.concat([pd.DataFrame(temp), pd.DataFrame(temp1), pd.DataFrame(temp2)], axis=1, ignore_index=False)
train = np.array(temp3)

temp = testPredict.reshape(test_size*area,1)
temp1 = testPredict1.reshape(test_size*area,1)
temp2 = test_Y.reshape(test_size*area,1)
temp3 = pd.concat([pd.DataFrame(temp), pd.DataFrame(temp1), pd.DataFrame(temp2)], axis=1, ignore_index=False)
test = np.array(temp3)


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(int(len(dataset)/look_back)):
        a = dataset[look_back*i:look_back*(i+1),0:dim]
        dataX.append(a)
        dataY.append(dataset[look_back*i:look_back*(i+1), dim])
    return np.array(dataX), np.array(dataY)

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape for LSTM  -> (size, timestep, dim) = (150, year, 99)
trainX_t, trainY_t = [],[]
testX_t, testY_t = [],[]

for k in range(train_size):
    temp_t = trainX[k].reshape(area, 1, dim)
    trainX_t.append(temp_t)
    target_t = trainY[k].reshape(area, 1)
    trainY_t.append(target_t)

for k in range(test_size):
    temp_t = testX[k].reshape(area, 1, dim)
    testX_t.append(temp_t)
    target_t = testY[k].reshape(area, 1)
    testY_t.append(target_t)


# build LSTM from keras
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import GRU
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import keras.backend as K
import time

lr = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
epoch = [10, 20, 30, 50,80,100, 150, 200, 400]
batch = [1,2,5,10,20,30,40,50, 75, 100, 150, 200]
hidden_size = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 150]
Result = []
Result = np.array(Result)

K.clear_session()
model = Sequential()  # Sequeatial Model
model.add(LSTM(150, input_shape=(1, dim), activation='hard_sigmoid'))  # (timestep, feature)
# model.add(GRU(150, input_shape=(1,dim), activation='hard_sigmoid')) # (timestep, feature)
model.add(Dense(1))  # output = 31
RMSprop = optimizers.RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='mean_absolute_percentage_error', optimizer='RMSprop', metrics=['accuracy'])
model.summary()

start = time.time()
for k in range(train_size):
    print(k)
    model.fit(trainX_t[k], trainY_t[k], epochs=30, verbose=2, batch_size=150)

# training
# trainPredict = model.predict(trainX_t)
# testPredict_rnn = model.predict(testX_t)

testPredict_rnn = []
testPredict_rnn = np.array(testPredict_rnn)
for k in range(test_size):
    print(k)
    temp = model.predict(testX_t[k])
    testPredict_rnn = np.append(testPredict_rnn, temp)
testPredict_rnn = testPredict_rnn.reshape(test_size, area)

end = time.time()

real_pred = []
real_pred = np.array(real_pred)
testY_t = np.array(testY_t).reshape(test_size, area)
for k in range(test_size):
    print(k)
    temp = testY_t[k] - testPredict_rnn[k]
    real_pred = np.append(real_pred, temp)
real_pred = real_pred.reshape(test_size, area)

RMSE = []
RMSE = np.array(RMSE)
for k in range(area):
    print(k)
    temp = (sum(real_pred[:, k] ** 2) / test_size) ** 0.5
    RMSE = np.append(RMSE, temp)

MAPE = []
MAPE = np.array(MAPE)
for k in range(area):
    print(k)
    temp = 100 * sum(abs(real_pred[:, k]) / testY_t[:, k]) / test_size
    MAPE = np.append(MAPE, temp)

temp = [np.mean(RMSE), np.ma.masked_invalid(MAPE).mean(), np.mean(MAPE),  (end - start) / 60]
print(temp)
        K.clear_session()
        model = Sequential() # Sequeatial Model
        model.add(LSTM(hidden_size[i], input_shape=(1,dim), activation='hard_sigmoid')) # (timestep, feature)
        # model.add(GRU(150, input_shape=(1,dim), activation='hard_sigmoid')) # (timestep, feature)
        model.add(Dense(1)) # output = 31
        RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
        model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])
        model.summary()

        start = time.time()
        for k in range(train_size):
            print(k)
            model.fit(trainX_t[k], trainY_t[k], epochs=100, verbose=2, batch_size=batch[j])

        # training
        #trainPredict = model.predict(trainX_t)
        # testPredict_rnn = model.predict(testX_t)

        testPredict_rnn = []
        testPredict_rnn = np.array(testPredict_rnn)
        for k in range(test_size):
            print(k)
            temp = model.predict(testX_t[k])
            testPredict_rnn = np.append(testPredict_rnn, temp)
        testPredict_rnn = testPredict_rnn.reshape(test_size, area)

        end = time.time()

        real_pred = []
        real_pred = np.array(real_pred)
        testY_t = np.array(testY_t).reshape(test_size, area)
        for k in range(test_size):
            print(k)
            temp = testY_t[k] - testPredict_rnn[k]
            real_pred = np.append(real_pred, temp)
        real_pred = real_pred.reshape(test_size, area)

        RMSE = []
        RMSE = np.array(RMSE)
        for k in range(area):
            print(k)
            temp =(sum(real_pred[:,k]**2)/test_size)**0.5
            RMSE = np.append(RMSE, temp)

        MAPE = []
        MAPE = np.array(MAPE)
        for k in range(area):
            print(k)
            temp = 100*sum(abs(real_pred[:,k])/testY_t[:,k])/test_size
            MAPE = np.append(MAPE, temp)

        temp = [hidden_size[i], batch[j], np.mean(RMSE), np.ma.masked_invalid(MAPE).mean(), (end-start)/60]

        Result = np.append(Result, temp)
        Result = Result.reshape(int(len(Result) / len(temp)), len(temp))

np.savetxt("/home/ysh1/PycharmProjects/thesis/results/time2/time2_hidden_batch.csv", Result, delimiter=",")
print(Result)

np.savetxt("/home/ysh1/PycharmProjects/thesis/results/comparison_remove/spatial_testPredict_rnn.csv", testPredict_rnn, delimiter=",")
np.savetxt("/home/ysh1/PycharmProjects/thesis/results/comparison_remove/spatial_RMSE.csv", RMSE, delimiter=",")
np.savetxt("/home/ysh1/PycharmProjects/thesis/results/comparison_remove/spatial_MAPE.csv", MAPE, delimiter=",")

# np.savetxt("/home/ysh1/PycharmProjects/thesis/results/grid/hidden_size/grid_testPredict_rnn_" + str(hidden_size[i])+".csv", testPredict_rnn, delimiter=",")
# np.savetxt("/home/ysh1/PycharmProjects/thesis/results/grid/hidden_size/grid_RMSE_" + str(hidden_size[i])+".csv", RMSE, delimiter=",")
# np.savetxt("/home/ysh1/PycharmProjects/thesis/results/grid/hidden_size/grid_MAPE_" + str(hidden_size[i])+".csv", MAPE, delimiter=",")

print(Result)

######## tuning - GCN ########
lr = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
epoch = [50,100,200,400,800,1000,1600,3200,5000]

layer = [1,2,4,5,8,10,12,15,20,40]
drop = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]


Result = []
Result = np.array(Result)

for i in range(len(layer)):
    for j in range(len(drop)):
        SYM_NORM = True
        num_filters = 1
        graph_conv_filters = preprocess_adj_tensor_with_identity(train_A, SYM_NORM)

        # build model
        X_input = Input(shape=(train_X.shape[1], train_X.shape[2]))
        # graph_conv_filters_input = Input(shape=(graph_conv_filters.shape[1], graph_conv_filters.shape[2]))
        graph_conv_filters_input = Input(shape=(train_A.shape[1], train_A.shape[2]))

        nb_epochs = 1000
        # hidden_size = 100
        # dim = 300
        batch_size = 7 #round(0.05*len(train_A))

        output = MultiGraphCNN(layer[i], num_filters, activation='relu')([X_input, graph_conv_filters_input])
        output = Dropout(drop[j])(output)
        output = MultiGraphCNN(layer[i], num_filters, activation='relu')([output, graph_conv_filters_input])
        output = Dropout(drop[j])(output)
        output = Lambda(lambda x: K.mean(x, axis=1))(output)  # adding a node invariant layer to make sure output does not depends upon the node order in a graph.
        output = Dense(train_Y.shape[1])(output)
        output = Activation('relu')(output)
        #
        # output = np.array(output)
        # output = output.reshape(None,1,300)
        # output = tf.convert_to_tensor(output)
        #
        # output = LSTM(hidden_size, input_shape=(1, dim), activation='hard_sigmoid')(output)
        # output = Dense(dim)(output)

        model = Model(inputs=[X_input, graph_conv_filters_input], outputs=output)

        # RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
        # sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
        # Adagrad = optimizers.Adagrad(learning_rate=0.01)
        adam = optimizers.Adam(learning_rate=0.2, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['acc'])
        # model.summary()

        import time
        start = time.time()
        # model.fit([train_X, graph_conv_filters], train_Y, batch_size=batch_size, epochs=nb_epochs, verbose=1)
        model.fit([train_X, train_A], train_Y, batch_size=batch_size, epochs=nb_epochs, verbose=1)

        # graph_conv_filters_test = preprocess_adj_tensor_with_identity(test_A, SYM_NORM)
        # testPredict = model.predict([test_X, graph_conv_filters_test])
        trainPredict = model.predict([train_X, train_A])
        testPredict = model.predict([test_X, test_A])


        # testScore = model.evaluate([test_X, graph_conv_filters_test], test_Y)
        testScore = model.evaluate([test_X, test_A], test_Y)
        testRMSE = math.sqrt(testScore[0])

        temp = [layer[i], drop[j], testRMSE, (time.time() - start)/60]
        Result = np.append(Result, temp)

print(Result)

# print(nb_epochs, "test_rmse: ", testRMSE)
# print("time: ", (time.time() - start)/60, "mins")


######## tuning - LSTM ########
# hidden_size = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]
# epoch = [10, 20, 50,80,100, 150, 200, 400]
epoch = 50
lr = 0.1
# lr = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
#batch = [168, 128, 84, 64, 42, 32, 21, 16, 10, 5]

Result = []
Result = np.array(Result)

for i in range(len(lr)):

    K.clear_session()
    model = Sequential() # Sequeatial Model
    model.add(LSTM(400, input_shape=(1,dim), activation='hard_sigmoid')) # (timestep, feature)
    model.add(Dense(1)) # output = 31
    RMSprop = optimizers.RMSprop(lr=lr[i], rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mean_absolute_percentage_error', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()
    start = time.time()
    for k in range(train_size):
        print(k)
        model.fit(trainX_t[k], trainY_t[k], epochs=epoch, verbose=2, batch_size=30)

    # training
    #trainPredict = model.predict(trainX_t)
    # testPredict_rnn = model.predict(testX_t)

    testPredict_rnn = []
    testPredict_rnn = np.array(testPredict_rnn)
    for k in range(test_size):
        print(k)
        temp = model.predict(testX_t[k])
        testPredict_rnn = np.append(testPredict_rnn, temp)
    testPredict_rnn = testPredict_rnn.reshape(test_size, area)

    # end = time.time()

    real_pred = []
    real_pred = np.array(real_pred)
    testY_t = np.array(testY_t).reshape(test_size, area)
    for k in range(test_size):
        print(k)
        temp = testY_t[k] - testPredict_rnn[k]
        real_pred = np.append(real_pred, temp)
    real_pred = real_pred.reshape(test_size, area)

    RMSE = []
    RMSE = np.array(RMSE)
    for k in range(area):
        print(k)
        temp = (sum(real_pred[:, k] ** 2) / test_size) ** 0.5
        RMSE = np.append(RMSE, temp)

    MAPE = []
    MAPE = np.array(MAPE)
    for k in range(area):
        print(k)
        temp = 100 * sum(abs(real_pred[:, k]) / testY_t[:, k]) / test_size
        MAPE = np.append(MAPE, temp)

    temp = [lr[i], np.mean(RMSE), np.mean(MAPE), time.time()-start]
    Result = np.append(Result, temp)

Result = Result.reshape(int(len(Result) / len(temp)), len(temp))
np.savetxt("/home/ysh1/PycharmProjects/thesis/lr_epoch_LSTM.csv", Result, delimiter=",")

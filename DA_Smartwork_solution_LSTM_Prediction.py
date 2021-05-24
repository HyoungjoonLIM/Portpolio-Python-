import numpy as np
import pandas as pd
from pandas import read_csv
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from math import gcd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

import tensorflow
import keras
### pip install versioned-hdf5 -> there was a problem @ installation of keras
### conda install tensorflow -> there was a problem @ installation of tf
from keras.utils import to_categorical
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.layers import Dense, ConvLSTM2D
from keras import optimizers, regularizers
import keras.backend as K

# load the dataset
def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = read_csv(filename, header=None, encoding='euc-kr')
    # retrieve numpy array
    dataset = data.values
    # split into input (X) and output (y) variables
    X = dataset[:, :-1]
    y = dataset[:, -1]
    # format all fields as string
    X = X.astype(str)
    # reshape target to be a 2d array
    y = y.reshape((len(y), 1))
    return X, y

# prepare input data
def prepare_inputs(X_train, X_test):
    oe = OrdinalEncoder()
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc


# prepare target
def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc

def lcms(arr):
    def lcm(x,y):
        return x*y//gcd(x,y)
    while True:
        arr.append(lcm(arr.pop(), arr.pop()))
        if len(arr) == 1:
            return arr[0]

# load the dataset
X, y = load_dataset('lstm_input_20_half2020.csv')

X = X[1:X.shape[0],:]
y = y[1:y.shape[0],:]

train_percentage = 0.9

train_size = int(X.shape[0] * train_percentage)+1
test_size = int(X.shape[0] * (1-train_percentage))

###################### Hyper-parameter tuning ###########################
Result = []
Result = np.array(Result)
# train parameters
timestep = seq = 1
dim = X.shape[1] + 1
hidden_size = [25, 50, 75, 100, 150, 200, 250, 300, 350, 400]
# hidden_size = 20
epoch = [25, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
lr = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
batch = [1024, 512, 400, 256, 128, 100, 64, 32, 16, 8]
look_back = area = 1

# build LSTM from keras

K.clear_session()
model = Sequential()  # Sequential Model
model.add(LSTM(75, input_shape=(None, 5), activation='hard_sigmoid'))
# model.add(SimpleRNN(40, activation='hard_sigmoid'))  # (timestep, feature)
model.add(Dense(5, activation='softmax'))  # output = 9 classes

adam = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train / test split
for fold in range(int(X.shape[0]/test_size)):

    X_test = X[test_size*fold:test_size*(fold+1)]
    y_test = y[test_size*fold:test_size*(fold+1)]

    if fold == 0:
        X_train = X[test_size:X.shape[0]]
        y_train = y[test_size:y.shape[0]]

    elif fold == (int(X.shape[0]/test_size)-1):
        X_train = X[0:train_size]
        y_train = y[0:train_size]

    else:
        X_train = np.append(X[0:test_size*fold], X[test_size*(fold+1):X.shape[0]])
        X_train = X_train.reshape(train_size,X.shape[1])
        y_train = np.append(y[0:test_size*fold], y[test_size*(fold+1):y.shape[0]])
        y_train = y_train.reshape(train_size,y.shape[1])
    print(fold)

    # prepare input data
    X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
    # prepare output data
    y_train_enc, y_test_enc = prepare_targets(y_train, y_test)

    # reshape for LSTM  -> (size, timestep, dim) = (150, year, 99)
    trainX_t, trainY_t = [],[]
    testX_t, testY_t = [],[]

    trainX_t = to_categorical(X_train_enc, num_classes = 5)
    testX_t = to_categorical(X_test_enc, num_classes = 5)
    trainY_t = to_categorical(y_train_enc, num_classes = 5)
    testY_t = to_categorical(y_test_enc, num_classes = 5)

    # trainY_t_integers = np.argmax(trainY_t, axis=1)
    # class_weights = compute_class_weight('balanced', np.unique(trainY_t_integers), trainY_t_integers)
    # d_class_weights = dict(enumerate(class_weights))

    d_class_weights = {0: 11, 1: 5, 2: 6.5, 3: 7, 4: 1}

    # model.summary()

    model.fit(trainX_t, trainY_t, epochs=100, verbose=2, batch_size=64, class_weight=d_class_weights)

    # training
    # trainPredict = model.predict(trainX_t)
    testPredict = model.predict_classes(testX_t)

    # testing
    # trainScore = model.evaluate(trainX_t, trainY_t)
    testScore = model.evaluate(testX_t, testY_t)
    F1score_M = f1_score(y_test_enc, testPredict, average='macro')
    F1score_m = f1_score(y_test_enc, testPredict, average='micro')
    F1score_w = f1_score(y_test_enc, testPredict, average='weighted')

    # print("train_rmse: ", trainRMSE)
    print("ACCURACY:", testScore[1] * 100, "%")
    print("F1score(macro):", F1score_M)
    print("F1score(micro):", F1score_m)

    ### for loop ###
    temp = [(fold+1), testScore[1] * 100, F1score_M, F1score_m]
    temp = np.array(temp)
    temp = temp.reshape(1, 4)
    Result = np.append(Result, temp)
    Result2 = Result.reshape((fold+1), 4)
    Result2 = pd.DataFrame(Result2)
    Result2.to_csv('result_crossval_weight_20_10fold.csv', index=False, encoding='cp949')




#####
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_percentage)
# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)

# reshape for LSTM  -> (size, timestep, dim) = (150, year, 99)
trainX_t, trainY_t = [], []
testX_t, testY_t = [], []

trainX_t = to_categorical(X_train_enc, num_classes=5)
testX_t = to_categorical(X_test_enc, num_classes=5)
trainY_t = to_categorical(y_train_enc, num_classes=5)
testY_t = to_categorical(y_test_enc, num_classes=5)

# trainY_t_integers = np.argmax(trainY_t, axis=1)
# class_weights = compute_class_weight('balanced', np.unique(trainY_t_integers), trainY_t_integers)
# d_class_weights = dict(enumerate(class_weights))

d_class_weights = {0: 11, 1: 5, 2: 6.5, 3: 7, 4: 1}

# training
# trainPredict = model.predict(trainX_t)
testPredict = model.predict_classes(testX_t)

# testing
# trainScore = model.evaluate(trainX_t, trainY_t)
testScore = model.evaluate(testX_t, testY_t)
F1score_M = f1_score(y_test_enc, testPredict, average='macro')
F1score_m = f1_score(y_test_enc, testPredict, average='micro')
cf = confusion_matrix(y_test_enc, testPredict)

# print("train_rmse: ", trainRMSE)
print(cf)
print("ACCURACY:", testScore[1] * 100, "%")
print("F1score(macro):", F1score_M)
print("F1score(micro):", F1score_m)


testPredict = pd.DataFrame(testPredict)
testPredict.to_csv('testPredict_20step_newweight2.csv', index=False, encoding='cp949')
y_test = pd.DataFrame(y_test)
y_test.to_csv('y_test_20step_newweight2.csv', index=False, encoding='euc-kr')
y_test_enc = pd.DataFrame(y_test_enc)
y_test_enc.to_csv('y_test_enc_20step_newweight2.csv', index=False, encoding='cp949')
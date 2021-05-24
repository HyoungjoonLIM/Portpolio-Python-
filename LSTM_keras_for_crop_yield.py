import numpy as np
import matplotlib.pyplot as plt
import math
import csv

###################### Hyper-parameter tuning ###########################
input = np.genfromtxt('/home/deeplearning2/PycharmProjects/LHJ/soybean/soy_apr_sep_16day_122_11x11.csv', delimiter=',')
#input = sorted(input, key=lambda input: input[0])
yr = 13
Result = []
Result = np.array(Result)

real_year = 2003 + yr
# train parameters
timestep = seq = 14
dim = len(input[1,:])-1
train_size = int(len(input) * 13/14)
test_size = int(len(input) * 1/14)
#hidden_size = [50, 100, 150, 200, 300, 400, 500, 600, 800, 1000]
hidden_size = 800
epoch = [30,40,50,60,70,80,90,100,125,150,200]
#lr = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
#batch = [168, 128, 84, 64, 42, 32, 21, 16, 10, 5]
look_back = area = 168

# train / test split
test = input[test_size*yr:test_size*(yr+1)]
if yr == 0:
    train = input[test_size:len(input)]
elif yr == 13:
    train = input[0:train_size]
else:
    train = np.append(input[0:test_size*yr], input[test_size*(yr+1):len(input)])
    train = train.reshape(train_size,len(input[1,:]))

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)/look_back):
        a = dataset[look_back*i:look_back*(i+1),1:dim]
        dataX.append(a)
        dataY.append(dataset[look_back*i:look_back*(i+1), dim])
    return np.array(dataX), np.array(dataY)

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape for LSTM  -> (size, timestep, dim) = (150, year, 99)
trainX_t, trainY_t = [],[]
testX_t, testY_t = [],[]

for j in range(seq-1):
    temp_t = trainX[j].reshape(area, 1, dim - 1)
    trainX_t.append(temp_t)
    target_t = trainY[j].reshape(area, 1)
    trainY_t.append(target_t)

testX_t = testX.reshape(area, 1, dim-1)
testY_t = testY.reshape(area, 1)

# build LSTM from keras
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import keras.backend as K

#for i in range(len(hidden_size)):
    for j in range(len(epoch)):

        K.clear_session()
        model = Sequential() # Sequeatial Model
        model.add(LSTM(hidden_size, input_shape=(1, dim-1), activation='hard_sigmoid')) # (timestep, feature)
        model.add(Dense(1)) # output = 1
        RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
        model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])
        model.summary()
        for k in range(seq-1):
            print(k)
            model.fit(trainX_t[k], trainY_t[k], epochs=epoch[j], verbose=2, batch_size=84)

        # training
        #trainPredict = model.predict(trainX_t)
        testPredict = model.predict(testX_t)

        # testing
        #trainScore = model.evaluate(trainX_t, trainY_t)
        testScore = model.evaluate(testX_t, testY_t)
        #trainRMSE = math.sqrt(trainScore[0])
        testRMSE = math.sqrt(testScore[0])
        testPRMSE = math.sqrt(testScore[0]) * 100 / np.mean(testY)

        #print("train_rmse: ", trainRMSE)
        print(hidden_size, epoch[j], "test_rmse: ", testRMSE)
        print(hidden_size, epoch[j], "test_%rmse: ", testPRMSE)

        temp = [hidden_size, epoch[j], testRMSE, testPRMSE]
        temp = np.array(temp)
        temp = temp.reshape(1,4)
        Result = np.append(Result, temp)

Result = Result.reshape(11,4)
print(Result)
prmse = Result[0:11,3]
#prmse = prmse.reshape(11,10)
print(prmse)

#####################################################################

input = np.genfromtxt('/home/deeplearning2/PycharmProjects/LHJ/rice/apr_aug_16day_99_11x9_90m.csv', delimiter=',')
#input = sorted(input, key=lambda input: input[0])
year = 14
Result15 = []
Result15 = np.array(Result15)

for yr in range(year):
    real_year = 2003 + yr
# train parameters
    timestep = seq = 14
    dim = len(input[1,:])-1
    train_size = int(len(input) * 13/14)
    test_size = int(len(input) * 1/14)
    hidden_size = 1000
    look_back = area = 150

    # train / test split
    test = input[test_size*yr:test_size*(yr+1)]
    if yr == 0:
        train = input[test_size:len(input)]
    elif yr == 13:
        train = input[0:train_size]
    else:
        train = np.append(input[0:test_size*yr], input[test_size*(yr+1):len(input)])
        train = train.reshape(train_size,len(input[1,:]))

    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)/look_back):
            a = dataset[look_back*i:look_back*(i+1),1:dim]
            dataX.append(a)
            dataY.append(dataset[look_back*i:look_back*(i+1), dim])
        return np.array(dataX), np.array(dataY)

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape for LSTM  -> (size, timestep, dim) = (150, year, 99)
    trainX_t, trainY_t = [],[]
    testX_t, testY_t = [],[]

    for j in range(seq-1):
        temp_t = trainX[j].reshape(area, 1, dim - 1)
        trainX_t.append(temp_t)
        target_t = trainY[j].reshape(area, 1)
        trainY_t.append(target_t)

    testX_t = testX.reshape(area, 1, dim-1)
    testY_t = testY.reshape(area, 1)

    # build LSTM from keras
    from keras.layers import LSTM
    from keras.models import Sequential
    from keras.layers import Dense
    from keras import optimizers
    import keras.backend as K

    K.clear_session()
    model = Sequential() # Sequeatial Model
    model.add(LSTM(hidden_size, input_shape=(1, dim-1), activation='hard_sigmoid')) # (timestep, feature)
    model.add(Dense(1)) # output = 1
    RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()
    for k in range(seq-1):
        print(k)
        model.fit(trainX_t[k], trainY_t[k], epochs=400, verbose=2, batch_size=75)

    # training
    #trainPredict = model.predict(trainX_t)
    testPredict = model.predict(testX_t)

    # testing
    #trainScore = model.evaluate(trainX_t, trainY_t)
    testScore = model.evaluate(testX_t, testY_t)
    #trainRMSE = math.sqrt(trainScore[0])
    testRMSE = math.sqrt(testScore[0])
    testPRMSE = math.sqrt(testScore[0]) * 100 / np.mean(testY)

    #print("train_rmse: ", trainRMSE)
    print("Year:" + str(real_year) + ", test_rmse: ", testRMSE)
    print("Year:" + str(real_year) + ", test_%rmse: ", testPRMSE)

    temp = [real_year, testRMSE, testPRMSE]
    temp = np.array(temp)
    temp = temp.reshape(1,3)
    Result15 = np.append(Result15, temp)

Result15 = Result15.reshape(14,3)
print(Result15)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result15[:,2]))

#plt.figure()
#plt.plot(testPredict)
#plt.plot(testY_t)

#####################################################################

input = np.genfromtxt('/home/deeplearning2/PycharmProjects/LHJ/rice/apr_sep_16day_121_11x11_90m.csv', delimiter=',')
#input = sorted(input, key=lambda input: input[0])
year = 14
Result16 = []
Result16 = np.array(Result16)

for yr in range(year):
    real_year = 2003 + yr
# train parameters
    timestep = seq = 14
    dim = len(input[1,:])-1
    train_size = int(len(input) * 13/14)
    test_size = int(len(input) * 1/14)
    hidden_size = 1000
    look_back = area = 150

    # train / test split
    test = input[test_size*yr:test_size*(yr+1)]
    if yr == 0:
        train = input[test_size:len(input)]
    elif yr == 13:
        train = input[0:train_size]
    else:
        train = np.append(input[0:test_size*yr], input[test_size*(yr+1):len(input)])
        train = train.reshape(train_size,len(input[1,:]))

    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)/look_back):
            a = dataset[look_back*i:look_back*(i+1),1:dim]
            dataX.append(a)
            dataY.append(dataset[look_back*i:look_back*(i+1), dim])
        return np.array(dataX), np.array(dataY)

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape for LSTM  -> (size, timestep, dim) = (150, year, 99)
    trainX_t, trainY_t = [],[]
    testX_t, testY_t = [],[]

    for j in range(seq-1):
        temp_t = trainX[j].reshape(area, 1, dim - 1)
        trainX_t.append(temp_t)
        target_t = trainY[j].reshape(area, 1)
        trainY_t.append(target_t)

    testX_t = testX.reshape(area, 1, dim-1)
    testY_t = testY.reshape(area, 1)

    # build LSTM from keras
    from keras.layers import LSTM
    from keras.models import Sequential
    from keras.layers import Dense
    import keras.backend as K

    K.clear_session()
    model = Sequential() # Sequeatial Model
    model.add(LSTM(hidden_size, input_shape=(1, dim-1), activation='hard_sigmoid')) # (timestep, feature)
    model.add(Dense(1)) # output = 1
    RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()
    for k in range(seq-1):
        print(k)
        model.fit(trainX_t[k], trainY_t[k], epochs=400, verbose=2, batch_size=84)
    # training
    #trainPredict = model.predict(trainX_t)
    testPredict = model.predict(testX_t)

    # testing
    #trainScore = model.evaluate(trainX_t, trainY_t)
    testScore = model.evaluate(testX_t, testY_t)
    #trainRMSE = math.sqrt(trainScore[0])
    testRMSE = math.sqrt(testScore[0])
    testPRMSE = math.sqrt(testScore[0]) * 100 / np.mean(testY)

    #print("train_rmse: ", trainRMSE)
    print("Year:" + str(real_year) + ", test_rmse: ", testRMSE)
    print("Year:" + str(real_year) + ", test_%rmse: ", testPRMSE)

    temp = [real_year, testRMSE, testPRMSE]
    temp = np.array(temp)
    temp = temp.reshape(1, 3)
    Result16 = np.append(Result16, temp)

Result16 = Result16.reshape(14, 3)
print(Result16)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result16[:,1]))

#plt.figure()
#plt.plot(testPredict)
#plt.plot(testY_t)

#####################################################################

input = np.genfromtxt('/home/deeplearning2/PycharmProjects/LHJ/rice/jun_aug_16day_55_11x5_90m.csv', delimiter=',')
#input = sorted(input, key=lambda input: input[0])
year = 14
Result17 = []
Result17 = np.array(Result17)

for yr in range(year):
    real_year = 2003 + yr
# train parameters
    timestep = seq = 14
    dim = len(input[1,:])-1
    train_size = int(len(input) * 13/14)
    test_size = int(len(input) * 1/14)
    hidden_size = 1000
    look_back = area = 150

    # train / test split
    test = input[test_size*yr:test_size*(yr+1)]
    if yr == 0:
        train = input[test_size:len(input)]
    elif yr == 13:
        train = input[0:train_size]
    else:
        train = np.append(input[0:test_size*yr], input[test_size*(yr+1):len(input)])
        train = train.reshape(train_size,len(input[1,:]))

    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)/look_back):
            a = dataset[look_back*i:look_back*(i+1),1:dim]
            dataX.append(a)
            dataY.append(dataset[look_back*i:look_back*(i+1), dim])
        return np.array(dataX), np.array(dataY)

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape for LSTM  -> (size, timestep, dim) = (150, year, 99)
    trainX_t, trainY_t = [],[]
    testX_t, testY_t = [],[]

    for j in range(seq-1):
        temp_t = trainX[j].reshape(area, 1, dim - 1)
        trainX_t.append(temp_t)
        target_t = trainY[j].reshape(area, 1)
        trainY_t.append(target_t)

    testX_t = testX.reshape(area, 1, dim-1)
    testY_t = testY.reshape(area, 1)

    # build LSTM from keras
    from keras.layers import LSTM
    from keras.models import Sequential
    from keras.layers import Dense
    import keras.backend as K

    K.clear_session()
    model = Sequential() # Sequeatial Model
    model.add(LSTM(hidden_size, input_shape=(1, dim-1), activation='hard_sigmoid')) # (timestep, feature)
    model.add(Dense(1)) # output = 1
    RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()
    for k in range(seq-1):
        print(k)
        model.fit(trainX_t[k], trainY_t[k], epochs=400, verbose=2, batch_size=84)

    # training
    #trainPredict = model.predict(trainX_t)
    testPredict = model.predict(testX_t)

    # testing
    #trainScore = model.evaluate(trainX_t, trainY_t)
    testScore = model.evaluate(testX_t, testY_t)
    #trainRMSE = math.sqrt(trainScore[0])
    testRMSE = math.sqrt(testScore[0])
    testPRMSE = math.sqrt(testScore[0]) * 100 / np.mean(testY)

    #print("train_rmse: ", trainRMSE)
    print("Year:" + str(real_year) + ", test_rmse: ", testRMSE)
    print("Year:" + str(real_year) + ", test_%rmse: ", testPRMSE)

    temp = [real_year, testRMSE, testPRMSE]
    temp = np.array(temp)
    temp = temp.reshape(1, 3)
    Result17 = np.append(Result17, temp)

Result17 = Result17.reshape(14, 3)
print(Result17)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result17[:,1]))

#plt.figure()
#plt.plot(testPredict)
#plt.plot(testY_t)

#####################################################################

input = np.genfromtxt('/home/deeplearning2/PycharmProjects/LHJ/rice/jun_sep_16day_77_11x7_90m.csv', delimiter=',')
#input = sorted(input, key=lambda input: input[0])
year = 14
Result18 = []
Result18 = np.array(Result18)

for yr in range(year):
    real_year = 2003 + yr
# train parameters
    timestep = seq = 14
    dim = len(input[1,:])-1
    train_size = int(len(input) * 13/14)
    test_size = int(len(input) * 1/14)
    hidden_size = 1000
    look_back = area = 150

    # train / test split
    test = input[test_size*yr:test_size*(yr+1)]
    if yr == 0:
        train = input[test_size:len(input)]
    elif yr == 13:
        train = input[0:train_size]
    else:
        train = np.append(input[0:test_size*yr], input[test_size*(yr+1):len(input)])
        train = train.reshape(train_size,len(input[1,:]))

    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)/look_back):
            a = dataset[look_back*i:look_back*(i+1),1:dim]
            dataX.append(a)
            dataY.append(dataset[look_back*i:look_back*(i+1), dim])
        return np.array(dataX), np.array(dataY)

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape for LSTM  -> (size, timestep, dim) = (150, year, 99)
    trainX_t, trainY_t = [],[]
    testX_t, testY_t = [],[]

    for j in range(seq-1):
        temp_t = trainX[j].reshape(area, 1, dim - 1)
        trainX_t.append(temp_t)
        target_t = trainY[j].reshape(area, 1)
        trainY_t.append(target_t)

    testX_t = testX.reshape(area, 1, dim-1)
    testY_t = testY.reshape(area, 1)

    # build LSTM from keras
    from keras.layers import LSTM
    from keras.models import Sequential
    from keras.layers import Dense
    import keras.backend as K

    K.clear_session()
    model = Sequential() # Sequeatial Model
    model.add(LSTM(hidden_size, input_shape=(1, dim-1), activation='hard_sigmoid')) # (timestep, feature)
    model.add(Dense(1)) # output = 1
    RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()
    for k in range(seq-1):
        print(k)
        model.fit(trainX_t[k], trainY_t[k], epochs=400, verbose=2, batch_size=84)

    # training
    #trainPredict = model.predict(trainX_t)
    testPredict = model.predict(testX_t)

    # testing
    #trainScore = model.evaluate(trainX_t, trainY_t)
    testScore = model.evaluate(testX_t, testY_t)
    #trainRMSE = math.sqrt(trainScore[0])
    testRMSE = math.sqrt(testScore[0])
    testPRMSE = math.sqrt(testScore[0]) * 100 / np.mean(testY)

    #print("train_rmse: ", trainRMSE)
    print("Year:" + str(real_year) + ", test_rmse: ", testRMSE)
    print("Year:" + str(real_year) + ", test_%rmse: ", testPRMSE)

    temp = [real_year, testRMSE, testPRMSE]
    temp = np.array(temp)
    temp = temp.reshape(1, 3)
    Result18 = np.append(Result18, temp)

Result18 = Result18.reshape(14, 3)
print(Result18)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result18[:,1]))

#plt.figure()
#plt.plot(testPredict)
#plt.plot(testY_t)

#####################################################################

input = np.genfromtxt('/home/deeplearning2/PycharmProjects/LHJ/rice/may_aug_16day_77_11x7_90m.csv', delimiter=',')
#input = sorted(input, key=lambda input: input[0])
year = 14
Result19 = []
Result19 = np.array(Result19)

for yr in range(year):
    real_year = 2003 + yr
# train parameters
    timestep = seq = 14
    dim = len(input[1,:])-1
    train_size = int(len(input) * 13/14)
    test_size = int(len(input) * 1/14)
    hidden_size = 1000
    look_back = area = 150

    # train / test split
    test = input[test_size*yr:test_size*(yr+1)]
    if yr == 0:
        train = input[test_size:len(input)]
    elif yr == 13:
        train = input[0:train_size]
    else:
        train = np.append(input[0:test_size*yr], input[test_size*(yr+1):len(input)])
        train = train.reshape(train_size,len(input[1,:]))

    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)/look_back):
            a = dataset[look_back*i:look_back*(i+1),1:dim]
            dataX.append(a)
            dataY.append(dataset[look_back*i:look_back*(i+1), dim])
        return np.array(dataX), np.array(dataY)

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape for LSTM  -> (size, timestep, dim) = (150, year, 99)
    trainX_t, trainY_t = [],[]
    testX_t, testY_t = [],[]

    for j in range(seq-1):
        temp_t = trainX[j].reshape(area, 1, dim - 1)
        trainX_t.append(temp_t)
        target_t = trainY[j].reshape(area, 1)
        trainY_t.append(target_t)

    testX_t = testX.reshape(area, 1, dim-1)
    testY_t = testY.reshape(area, 1)

    # build LSTM from keras
    from keras.layers import LSTM
    from keras.models import Sequential
    from keras.layers import Dense
    from keras import optimizers
    import keras.backend as K

    K.clear_session()
    model = Sequential() # Sequeatial Model
    model.add(LSTM(hidden_size, input_shape=(1, dim-1), activation='hard_sigmoid')) # (timestep, feature)
    model.add(Dense(1)) # output = 1
    RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()
    for k in range(seq-1):
        print(k)
        model.fit(trainX_t[k], trainY_t[k], epochs=400, verbose=2, batch_size=84)

    # training
    #trainPredict = model.predict(trainX_t)
    testPredict = model.predict(testX_t)

    # testing
    #trainScore = model.evaluate(trainX_t, trainY_t)
    testScore = model.evaluate(testX_t, testY_t)
    #trainRMSE = math.sqrt(trainScore[0])
    testRMSE = math.sqrt(testScore[0])
    testPRMSE = math.sqrt(testScore[0]) * 100 / np.mean(testY)

    #print("train_rmse: ", trainRMSE)
    print("Year:" + str(real_year) + ", test_rmse: ", testRMSE)
    print("Year:" + str(real_year) + ", test_%rmse: ", testPRMSE)

    temp = [real_year, testRMSE, testPRMSE]
    temp = np.array(temp)
    temp = temp.reshape(1, 3)
    Result19 = np.append(Result19, temp)

Result19 = Result19.reshape(14, 3)
print(Result19)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result19[:,1]))

#plt.figure()
#plt.plot(testPredict)
#plt.plot(testY_t)

#####################################################################

input = np.genfromtxt('/home/deeplearning2/PycharmProjects/LHJ/rice/may_sep_16day_99_11x9_90m.csv', delimiter=',')
#input = sorted(input, key=lambda input: input[0])
year = 14
Result20 = []
Result20 = np.array(Result20)

for yr in range(year):
    real_year = 2003 + yr
# train parameters
    timestep = seq = 14
    dim = len(input[1,:])-1
    train_size = int(len(input) * 13/14)
    test_size = int(len(input) * 1/14)
    hidden_size = 1000
    look_back = area = 150

    # train / test split
    test = input[test_size*yr:test_size*(yr+1)]
    if yr == 0:
        train = input[test_size:len(input)]
    elif yr == 13:
        train = input[0:train_size]
    else:
        train = np.append(input[0:test_size*yr], input[test_size*(yr+1):len(input)])
        train = train.reshape(train_size,len(input[1,:]))

    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)/look_back):
            a = dataset[look_back*i:look_back*(i+1),1:dim]
            dataX.append(a)
            dataY.append(dataset[look_back*i:look_back*(i+1), dim])
        return np.array(dataX), np.array(dataY)

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape for LSTM  -> (size, timestep, dim) = (150, year, 99)
    trainX_t, trainY_t = [],[]
    testX_t, testY_t = [],[]

    for j in range(seq-1):
        temp_t = trainX[j].reshape(area, 1, dim - 1)
        trainX_t.append(temp_t)
        target_t = trainY[j].reshape(area, 1)
        trainY_t.append(target_t)

    testX_t = testX.reshape(area, 1, dim-1)
    testY_t = testY.reshape(area, 1)

    # build LSTM from keras
    from keras.layers import LSTM
    from keras.models import Sequential
    from keras.layers import Dense
    import keras.backend as K

    K.clear_session()
    model = Sequential() # Sequeatial Model
    model.add(LSTM(hidden_size, input_shape=(1, dim-1), activation='hard_sigmoid')) # (timestep, feature)
    model.add(Dense(1)) # output = 1
    RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()
    for k in range(seq-1):
        print(k)
        model.fit(trainX_t[k], trainY_t[k], epochs=400, verbose=2, batch_size=84)

    # training
    #trainPredict = model.predict(trainX_t)
    testPredict = model.predict(testX_t)

    # testing
    #trainScore = model.evaluate(trainX_t, trainY_t)
    testScore = model.evaluate(testX_t, testY_t)
    #trainRMSE = math.sqrt(trainScore[0])
    testRMSE = math.sqrt(testScore[0])
    testPRMSE = math.sqrt(testScore[0]) * 100 / np.mean(testY)

    #print("train_rmse: ", trainRMSE)
    print("Year:" + str(real_year) + ", test_rmse: ", testRMSE)
    print("Year:" + str(real_year) + ", test_%rmse: ", testPRMSE)

    temp = [real_year, testRMSE, testPRMSE]
    temp = np.array(temp)
    temp = temp.reshape(1, 3)
    Result20 = np.append(Result20, temp)

Result20 = Result20.reshape(14, 3)
print(Result20)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result20[:,1]))

#plt.figure()
#plt.plot(testPredict)
#plt.plot(testY_t)

#####################################################################

input = np.genfromtxt('/home/deeplearning2/PycharmProjects/LHJ/corn/corn_apr_aug_16day_100_11x9.csv', delimiter=',')
#input = sorted(input, key=lambda input: input[0])
year = 14
Result3 = []
Result3 = np.array(Result3)

for yr in range(year):
    real_year = 2003 + yr
# train parameters
    timestep = seq = 14
    dim = len(input[1,:])-1
    train_size = int(len(input) * 13/14)
    test_size = int(len(input) * 1/14)
    hidden_size = 425
    look_back = area = 168

    # train / test split
    test = input[test_size*yr:test_size*(yr+1)]
    if yr == 0:
        train = input[test_size:len(input)]
    elif yr == 13:
        train = input[0:train_size]
    else:
        train = np.append(input[0:test_size*yr], input[test_size*(yr+1):len(input)])
        train = train.reshape(train_size,len(input[1,:]))

    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)/look_back):
            a = dataset[look_back*i:look_back*(i+1),1:dim]
            dataX.append(a)
            dataY.append(dataset[look_back*i:look_back*(i+1), dim])
        return np.array(dataX), np.array(dataY)

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape for LSTM  -> (size, timestep, dim) = (150, year, 99)
    trainX_t, trainY_t = [],[]
    testX_t, testY_t = [],[]

    for j in range(seq-1):
        temp_t = trainX[j].reshape(area, 1, dim - 1)
        trainX_t.append(temp_t)
        target_t = trainY[j].reshape(area, 1)
        trainY_t.append(target_t)

    testX_t = testX.reshape(area, 1, dim-1)
    testY_t = testY.reshape(area, 1)

    # build LSTM from keras
    from keras.layers import LSTM
    from keras.models import Sequential
    from keras.layers import Dense
    from keras import optimizers
    import keras.backend as K

    K.clear_session()
    model = Sequential() # Sequeatial Model
    model.add(LSTM(hidden_size, input_shape=(1, dim-1), activation='hard_sigmoid')) # (timestep, feature)
    model.add(Dense(1)) # output = 1
    RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()
    for k in range(seq-1):
        print(k)
        model.fit(trainX_t[k], trainY_t[k], epochs=80, verbose=2, batch_size=84)

    # training
    #trainPredict = model.predict(trainX_t)
    testPredict = model.predict(testX_t)

    # testing
    #trainScore = model.evaluate(trainX_t, trainY_t)
    testScore = model.evaluate(testX_t, testY_t)
    #trainRMSE = math.sqrt(trainScore[0])
    testRMSE = math.sqrt(testScore[0])
    testPRMSE = math.sqrt(testScore[0]) * 100 / np.mean(testY)

    #print("train_rmse: ", trainRMSE)
    print("Year:" + str(real_year) + ", test_rmse: ", testRMSE)
    print("Year:" + str(real_year) + ", test_%rmse: ", testPRMSE)

    temp = [real_year, testRMSE, testPRMSE]
    temp = np.array(temp)
    temp = temp.reshape(1, 3)
    Result3 = np.append(Result3, temp)

Result3 = Result3.reshape(14, 3)
print(Result3)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result3[:,1]))

#plt.figure()
#plt.plot(testPredict)
#plt.plot(testY_t)

#####################################################################

input = np.genfromtxt('/home/deeplearning2/PycharmProjects/LHJ/corn/corn_apr_sep_16day_122_11x11.csv', delimiter=',')
#input = sorted(input, key=lambda input: input[0])
year = 14
Result4 = []
Result4 = np.array(Result4)

for yr in range(year):
    real_year = 2003 + yr
# train parameters
    timestep = seq = 14
    dim = len(input[1,:])-1
    train_size = int(len(input) * 13/14)
    test_size = int(len(input) * 1/14)
    hidden_size = 425
    look_back = area = 168

    # train / test split
    test = input[test_size*yr:test_size*(yr+1)]
    if yr == 0:
        train = input[test_size:len(input)]
    elif yr == 13:
        train = input[0:train_size]
    else:
        train = np.append(input[0:test_size*yr], input[test_size*(yr+1):len(input)])
        train = train.reshape(train_size,len(input[1,:]))

    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)/look_back):
            a = dataset[look_back*i:look_back*(i+1),1:dim]
            dataX.append(a)
            dataY.append(dataset[look_back*i:look_back*(i+1), dim])
        return np.array(dataX), np.array(dataY)

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape for LSTM  -> (size, timestep, dim) = (150, year, 99)
    trainX_t, trainY_t = [],[]
    testX_t, testY_t = [],[]

    for j in range(seq-1):
        temp_t = trainX[j].reshape(area, 1, dim - 1)
        trainX_t.append(temp_t)
        target_t = trainY[j].reshape(area, 1)
        trainY_t.append(target_t)

    testX_t = testX.reshape(area, 1, dim-1)
    testY_t = testY.reshape(area, 1)

    # build LSTM from keras
    from keras.layers import LSTM
    from keras.models import Sequential
    from keras.layers import Dense
    import keras.backend as K

    K.clear_session()
    model = Sequential() # Sequeatial Model
    model.add(LSTM(hidden_size, input_shape=(1, dim-1), activation='hard_sigmoid')) # (timestep, feature)
    model.add(Dense(1)) # output = 1
    RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()
    for k in range(seq-1):
        print(k)
        model.fit(trainX_t[k], trainY_t[k], epochs=80, verbose=2, batch_size=84)
    # training
    #trainPredict = model.predict(trainX_t)
    testPredict = model.predict(testX_t)

    # testing
    #trainScore = model.evaluate(trainX_t, trainY_t)
    testScore = model.evaluate(testX_t, testY_t)
    #trainRMSE = math.sqrt(trainScore[0])
    testRMSE = math.sqrt(testScore[0])
    testPRMSE = math.sqrt(testScore[0]) * 100 / np.mean(testY)

    #print("train_rmse: ", trainRMSE)
    print("Year:" + str(real_year) + ", test_rmse: ", testRMSE)
    print("Year:" + str(real_year) + ", test_%rmse: ", testPRMSE)

    temp = [real_year, testRMSE, testPRMSE]
    temp = np.array(temp)
    temp = temp.reshape(1, 3)
    Result4 = np.append(Result4, temp)

Result4 = Result4.reshape(14, 3)
print(Result4)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result4[:,1]))

#plt.figure()
#plt.plot(testPredict)
#plt.plot(testY_t)

#####################################################################

input = np.genfromtxt('/home/deeplearning2/PycharmProjects/LHJ/corn/corn_jun_aug_16day_56_11x5.csv', delimiter=',')
#input = sorted(input, key=lambda input: input[0])
year = 14
Result5 = []
Result5 = np.array(Result5)

for yr in range(year):
    real_year = 2003 + yr
# train parameters
    timestep = seq = 14
    dim = len(input[1,:])-1
    train_size = int(len(input) * 13/14)
    test_size = int(len(input) * 1/14)
    hidden_size = 425
    look_back = area = 168

    # train / test split
    test = input[test_size*yr:test_size*(yr+1)]
    if yr == 0:
        train = input[test_size:len(input)]
    elif yr == 13:
        train = input[0:train_size]
    else:
        train = np.append(input[0:test_size*yr], input[test_size*(yr+1):len(input)])
        train = train.reshape(train_size,len(input[1,:]))

    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)/look_back):
            a = dataset[look_back*i:look_back*(i+1),1:dim]
            dataX.append(a)
            dataY.append(dataset[look_back*i:look_back*(i+1), dim])
        return np.array(dataX), np.array(dataY)

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape for LSTM  -> (size, timestep, dim) = (150, year, 99)
    trainX_t, trainY_t = [],[]
    testX_t, testY_t = [],[]

    for j in range(seq-1):
        temp_t = trainX[j].reshape(area, 1, dim - 1)
        trainX_t.append(temp_t)
        target_t = trainY[j].reshape(area, 1)
        trainY_t.append(target_t)

    testX_t = testX.reshape(area, 1, dim-1)
    testY_t = testY.reshape(area, 1)

    # build LSTM from keras
    from keras.layers import LSTM
    from keras.models import Sequential
    from keras.layers import Dense
    import keras.backend as K

    K.clear_session()
    model = Sequential() # Sequeatial Model
    model.add(LSTM(hidden_size, input_shape=(1, dim-1), activation='hard_sigmoid')) # (timestep, feature)
    model.add(Dense(1)) # output = 1
    RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()
    for k in range(seq-1):
        print(k)
        model.fit(trainX_t[k], trainY_t[k], epochs=80, verbose=2, batch_size=84)

    # training
    #trainPredict = model.predict(trainX_t)
    testPredict = model.predict(testX_t)

    # testing
    #trainScore = model.evaluate(trainX_t, trainY_t)
    testScore = model.evaluate(testX_t, testY_t)
    #trainRMSE = math.sqrt(trainScore[0])
    testRMSE = math.sqrt(testScore[0])
    testPRMSE = math.sqrt(testScore[0]) * 100 / np.mean(testY)

    #print("train_rmse: ", trainRMSE)
    print("Year:" + str(real_year) + ", test_rmse: ", testRMSE)
    print("Year:" + str(real_year) + ", test_%rmse: ", testPRMSE)

    temp = [real_year, testRMSE, testPRMSE]
    temp = np.array(temp)
    temp = temp.reshape(1, 3)
    Result5 = np.append(Result5, temp)

Result5 = Result5.reshape(14, 3)
print(Result5)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result5[:,1]))

#plt.figure()
#plt.plot(testPredict)
#plt.plot(testY_t)

#####################################################################

input = np.genfromtxt('/home/deeplearning2/PycharmProjects/LHJ/corn/corn_jun_sep_16day_78_11x7.csv', delimiter=',')
#input = sorted(input, key=lambda input: input[0])
year = 14
Result6 = []
Result6 = np.array(Result6)

for yr in range(year):
    real_year = 2003 + yr
# train parameters
    timestep = seq = 14
    dim = len(input[1,:])-1
    train_size = int(len(input) * 13/14)
    test_size = int(len(input) * 1/14)
    hidden_size = 425
    look_back = area = 168

    # train / test split
    test = input[test_size*yr:test_size*(yr+1)]
    if yr == 0:
        train = input[test_size:len(input)]
    elif yr == 13:
        train = input[0:train_size]
    else:
        train = np.append(input[0:test_size*yr], input[test_size*(yr+1):len(input)])
        train = train.reshape(train_size,len(input[1,:]))

    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)/look_back):
            a = dataset[look_back*i:look_back*(i+1),1:dim]
            dataX.append(a)
            dataY.append(dataset[look_back*i:look_back*(i+1), dim])
        return np.array(dataX), np.array(dataY)

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape for LSTM  -> (size, timestep, dim) = (150, year, 99)
    trainX_t, trainY_t = [],[]
    testX_t, testY_t = [],[]

    for j in range(seq-1):
        temp_t = trainX[j].reshape(area, 1, dim - 1)
        trainX_t.append(temp_t)
        target_t = trainY[j].reshape(area, 1)
        trainY_t.append(target_t)

    testX_t = testX.reshape(area, 1, dim-1)
    testY_t = testY.reshape(area, 1)

    # build LSTM from keras
    from keras.layers import LSTM
    from keras.models import Sequential
    from keras.layers import Dense
    import keras.backend as K

    K.clear_session()
    model = Sequential() # Sequeatial Model
    model.add(LSTM(hidden_size, input_shape=(1, dim-1), activation='hard_sigmoid')) # (timestep, feature)
    model.add(Dense(1)) # output = 1
    RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()
    for k in range(seq-1):
        print(k)
        model.fit(trainX_t[k], trainY_t[k], epochs=80, verbose=2, batch_size=84)

    # training
    #trainPredict = model.predict(trainX_t)
    testPredict = model.predict(testX_t)

    # testing
    #trainScore = model.evaluate(trainX_t, trainY_t)
    testScore = model.evaluate(testX_t, testY_t)
    #trainRMSE = math.sqrt(trainScore[0])
    testRMSE = math.sqrt(testScore[0])
    testPRMSE = math.sqrt(testScore[0]) * 100 / np.mean(testY)

    #print("train_rmse: ", trainRMSE)
    print("Year:" + str(real_year) + ", test_rmse: ", testRMSE)
    print("Year:" + str(real_year) + ", test_%rmse: ", testPRMSE)

    temp = [real_year, testRMSE, testPRMSE]
    temp = np.array(temp)
    temp = temp.reshape(1, 3)
    Result6 = np.append(Result6, temp)

Result6 = Result6.reshape(14, 3)
print(Result6)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result6[:,1]))

#plt.figure()
#plt.plot(testPredict)
#plt.plot(testY_t)

#####################################################################

input = np.genfromtxt('/home/deeplearning2/PycharmProjects/LHJ/corn/corn_may_aug_16day_78_11x7.csv', delimiter=',')
#input = sorted(input, key=lambda input: input[0])
year = 14
Result7 = []
Result7 = np.array(Result7)

for yr in range(year):
    real_year = 2003 + yr
# train parameters
    timestep = seq = 14
    dim = len(input[1,:])-1
    train_size = int(len(input) * 13/14)
    test_size = int(len(input) * 1/14)
    hidden_size = 425
    look_back = area = 168

    # train / test split
    test = input[test_size*yr:test_size*(yr+1)]
    if yr == 0:
        train = input[test_size:len(input)]
    elif yr == 13:
        train = input[0:train_size]
    else:
        train = np.append(input[0:test_size*yr], input[test_size*(yr+1):len(input)])
        train = train.reshape(train_size,len(input[1,:]))

    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)/look_back):
            a = dataset[look_back*i:look_back*(i+1),1:dim]
            dataX.append(a)
            dataY.append(dataset[look_back*i:look_back*(i+1), dim])
        return np.array(dataX), np.array(dataY)

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape for LSTM  -> (size, timestep, dim) = (150, year, 99)
    trainX_t, trainY_t = [],[]
    testX_t, testY_t = [],[]

    for j in range(seq-1):
        temp_t = trainX[j].reshape(area, 1, dim - 1)
        trainX_t.append(temp_t)
        target_t = trainY[j].reshape(area, 1)
        trainY_t.append(target_t)

    testX_t = testX.reshape(area, 1, dim-1)
    testY_t = testY.reshape(area, 1)

    # build LSTM from keras
    from keras.layers import LSTM
    from keras.models import Sequential
    from keras.layers import Dense
    from keras import optimizers
    import keras.backend as K

    K.clear_session()
    model = Sequential() # Sequeatial Model
    model.add(LSTM(hidden_size, input_shape=(1, dim-1), activation='hard_sigmoid')) # (timestep, feature)
    model.add(Dense(1)) # output = 1
    RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()
    for k in range(seq-1):
        print(k)
        model.fit(trainX_t[k], trainY_t[k], epochs=80, verbose=2, batch_size=84)

    # training
    #trainPredict = model.predict(trainX_t)
    testPredict = model.predict(testX_t)

    # testing
    #trainScore = model.evaluate(trainX_t, trainY_t)
    testScore = model.evaluate(testX_t, testY_t)
    #trainRMSE = math.sqrt(trainScore[0])
    testRMSE = math.sqrt(testScore[0])
    testPRMSE = math.sqrt(testScore[0]) * 100 / np.mean(testY)

    #print("train_rmse: ", trainRMSE)
    print("Year:" + str(real_year) + ", test_rmse: ", testRMSE)
    print("Year:" + str(real_year) + ", test_%rmse: ", testPRMSE)

    temp = [real_year, testRMSE, testPRMSE]
    temp = np.array(temp)
    temp = temp.reshape(1, 3)
    Result7 = np.append(Result7, temp)

Result7 = Result7.reshape(14, 3)
print(Result7)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result7[:,1]))

#plt.figure()
#plt.plot(testPredict)
#plt.plot(testY_t)

#####################################################################

input = np.genfromtxt('/home/deeplearning2/PycharmProjects/LHJ/corn/corn_may_sep_16day_100_11x9.csv', delimiter=',')
#input = sorted(input, key=lambda input: input[0])
year = 14
Result8 = []
Result8 = np.array(Result8)

for yr in range(year):
    real_year = 2003 + yr
# train parameters
    timestep = seq = 14
    dim = len(input[1,:])-1
    train_size = int(len(input) * 13/14)
    test_size = int(len(input) * 1/14)
    hidden_size = 425
    look_back = area = 168

    # train / test split
    test = input[test_size*yr:test_size*(yr+1)]
    if yr == 0:
        train = input[test_size:len(input)]
    elif yr == 13:
        train = input[0:train_size]
    else:
        train = np.append(input[0:test_size*yr], input[test_size*(yr+1):len(input)])
        train = train.reshape(train_size,len(input[1,:]))

    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)/look_back):
            a = dataset[look_back*i:look_back*(i+1),1:dim]
            dataX.append(a)
            dataY.append(dataset[look_back*i:look_back*(i+1), dim])
        return np.array(dataX), np.array(dataY)

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape for LSTM  -> (size, timestep, dim) = (150, year, 99)
    trainX_t, trainY_t = [],[]
    testX_t, testY_t = [],[]

    for j in range(seq-1):
        temp_t = trainX[j].reshape(area, 1, dim - 1)
        trainX_t.append(temp_t)
        target_t = trainY[j].reshape(area, 1)
        trainY_t.append(target_t)

    testX_t = testX.reshape(area, 1, dim-1)
    testY_t = testY.reshape(area, 1)

    # build LSTM from keras
    from keras.layers import LSTM
    from keras.models import Sequential
    from keras.layers import Dense
    import keras.backend as K

    K.clear_session()
    model = Sequential() # Sequeatial Model
    model.add(LSTM(hidden_size, input_shape=(1, dim-1), activation='hard_sigmoid')) # (timestep, feature)
    model.add(Dense(1)) # output = 1
    RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()
    for k in range(seq-1):
        print(k)
        model.fit(trainX_t[k], trainY_t[k], epochs=80, verbose=2, batch_size=84)

    # training
    #trainPredict = model.predict(trainX_t)
    testPredict = model.predict(testX_t)

    # testing
    #trainScore = model.evaluate(trainX_t, trainY_t)
    testScore = model.evaluate(testX_t, testY_t)
    #trainRMSE = math.sqrt(trainScore[0])
    testRMSE = math.sqrt(testScore[0])
    testPRMSE = math.sqrt(testScore[0]) * 100 / np.mean(testY)

    #print("train_rmse: ", trainRMSE)
    print("Year:" + str(real_year) + ", test_rmse: ", testRMSE)
    print("Year:" + str(real_year) + ", test_%rmse: ", testPRMSE)

    temp = [real_year, testRMSE, testPRMSE]
    temp = np.array(temp)
    temp = temp.reshape(1, 3)
    Result8 = np.append(Result8, temp)

Result8 = Result8.reshape(14, 3)
print(Result8)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result8[:,1]))

#plt.figure()
#plt.plot(testPredict)
#plt.plot(testY_t)


#####################################################################
input = np.genfromtxt('/home/deeplearning2/PycharmProjects/LHJ/soybean/soy_apr_aug_16day_100_11x9.csv', delimiter=',')
#input = sorted(input, key=lambda input: input[0])
year = 14
Result9 = []
Result9 = np.array(Result9)

for yr in range(year):
    real_year = 2003 + yr
# train parameters
    timestep = seq = 14
    dim = len(input[1,:])-1
    train_size = int(len(input) * 13/14)
    test_size = int(len(input) * 1/14)
    hidden_size = 800
    look_back = area = 168

    # train / test split
    test = input[test_size*yr:test_size*(yr+1)]
    if yr == 0:
        train = input[test_size:len(input)]
    elif yr == 13:
        train = input[0:train_size]
    else:
        train = np.append(input[0:test_size*yr], input[test_size*(yr+1):len(input)])
        train = train.reshape(train_size,len(input[1,:]))

    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)/look_back):
            a = dataset[look_back*i:look_back*(i+1),1:dim]
            dataX.append(a)
            dataY.append(dataset[look_back*i:look_back*(i+1), dim])
        return np.array(dataX), np.array(dataY)

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape for LSTM  -> (size, timestep, dim) = (150, year, 99)
    trainX_t, trainY_t = [],[]
    testX_t, testY_t = [],[]

    for j in range(seq-1):
        temp_t = trainX[j].reshape(area, 1, dim - 1)
        trainX_t.append(temp_t)
        target_t = trainY[j].reshape(area, 1)
        trainY_t.append(target_t)

    testX_t = testX.reshape(area, 1, dim-1)
    testY_t = testY.reshape(area, 1)

    # build LSTM from keras
    from keras.layers import LSTM
    from keras.models import Sequential
    from keras.layers import Dense
    from keras import optimizers
    import keras.backend as K

    K.clear_session()
    model = Sequential() # Sequeatial Model
    model.add(LSTM(hidden_size, input_shape=(1, dim-1), activation='hard_sigmoid')) # (timestep, feature)
    model.add(Dense(1)) # output = 1
    RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()
    for k in range(seq-1):
        print(k)
        model.fit(trainX_t[k], trainY_t[k], epochs=200, verbose=2, batch_size=84)

    # training
    #trainPredict = model.predict(trainX_t)
    testPredict = model.predict(testX_t)

    # testing
    #trainScore = model.evaluate(trainX_t, trainY_t)
    testScore = model.evaluate(testX_t, testY_t)
    #trainRMSE = math.sqrt(trainScore[0])
    testRMSE = math.sqrt(testScore[0])
    testPRMSE = math.sqrt(testScore[0]) * 100 / np.mean(testY)

    #print("train_rmse: ", trainRMSE)
    print("Year:" + str(real_year) + ", test_rmse: ", testRMSE)
    print("Year:" + str(real_year) + ", test_%rmse: ", testPRMSE)

    temp = [real_year, testRMSE, testPRMSE]
    temp = np.array(temp)
    temp = temp.reshape(1, 3)
    Result9 = np.append(Result9, temp)

Result9 = Result9.reshape(14, 3)
print(Result9)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result9[:,2]))

#plt.figure()
#plt.plot(testPredict)
#plt.plot(testY_t)

#####################################################################

input = np.genfromtxt('/home/deeplearning2/PycharmProjects/LHJ/soybean/soy_apr_sep_16day_122_11x11.csv', delimiter=',')
#input = sorted(input, key=lambda input: input[0])
year = 14
Result10 = []
Result10 = np.array(Result10)

for yr in range(year):
    real_year = 2003 + yr
# train parameters
    timestep = seq = 14
    dim = len(input[1,:])-1
    train_size = int(len(input) * 13/14)
    test_size = int(len(input) * 1/14)
    hidden_size = 800
    look_back = area = 168

    # train / test split
    test = input[test_size*yr:test_size*(yr+1)]
    if yr == 0:
        train = input[test_size:len(input)]
    elif yr == 13:
        train = input[0:train_size]
    else:
        train = np.append(input[0:test_size*yr], input[test_size*(yr+1):len(input)])
        train = train.reshape(train_size,len(input[1,:]))

    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)/look_back):
            a = dataset[look_back*i:look_back*(i+1),1:dim]
            dataX.append(a)
            dataY.append(dataset[look_back*i:look_back*(i+1), dim])
        return np.array(dataX), np.array(dataY)

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape for LSTM  -> (size, timestep, dim) = (150, year, 99)
    trainX_t, trainY_t = [],[]
    testX_t, testY_t = [],[]

    for j in range(seq-1):
        temp_t = trainX[j].reshape(area, 1, dim - 1)
        trainX_t.append(temp_t)
        target_t = trainY[j].reshape(area, 1)
        trainY_t.append(target_t)

    testX_t = testX.reshape(area, 1, dim-1)
    testY_t = testY.reshape(area, 1)

    # build LSTM from keras
    from keras.layers import LSTM
    from keras.models import Sequential
    from keras.layers import Dense
    import keras.backend as K

    K.clear_session()
    model = Sequential()  # Sequeatial Model
    model.add(LSTM(hidden_size, input_shape=(1, dim - 1), activation='hard_sigmoid'))  # (timestep, feature)
    model.add(Dense(1))  # output = 1
    RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()
    for k in range(seq - 1):
        print(k)
        model.fit(trainX_t[k], trainY_t[k], epochs=200, verbose=2, batch_size=84)
    # training
    #trainPredict = model.predict(trainX_t)
    testPredict = model.predict(testX_t)

    # testing
    #trainScore = model.evaluate(trainX_t, trainY_t)
    testScore = model.evaluate(testX_t, testY_t)
    #trainRMSE = math.sqrt(trainScore[0])
    testRMSE = math.sqrt(testScore[0])
    testPRMSE = math.sqrt(testScore[0]) * 100 / np.mean(testY)

    #print("train_rmse: ", trainRMSE)
    print("Year:" + str(real_year) + ", test_rmse: ", testRMSE)
    print("Year:" + str(real_year) + ", test_%rmse: ", testPRMSE)

    temp = [real_year, testRMSE, testPRMSE]
    temp = np.array(temp)
    temp = temp.reshape(1, 3)
    Result10 = np.append(Result10, temp)

Result10 = Result10.reshape(14, 3)
print(Result10)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result10[:,2]))

#plt.figure()
#plt.plot(testPredict)
#plt.plot(testY_t)

#####################################################################

input = np.genfromtxt('/home/deeplearning2/PycharmProjects/LHJ/soybean/soy_jun_aug_16day_56_11x5.csv', delimiter=',')
#input = sorted(input, key=lambda input: input[0])
year = 14
Result11 = []
Result11 = np.array(Result11)

for yr in range(year):
    real_year = 2003 + yr
# train parameters
    timestep = seq = 14
    dim = len(input[1,:])-1
    train_size = int(len(input) * 13/14)
    test_size = int(len(input) * 1/14)
    hidden_size = 800
    look_back = area = 168

    # train / test split
    test = input[test_size*yr:test_size*(yr+1)]
    if yr == 0:
        train = input[test_size:len(input)]
    elif yr == 13:
        train = input[0:train_size]
    else:
        train = np.append(input[0:test_size*yr], input[test_size*(yr+1):len(input)])
        train = train.reshape(train_size,len(input[1,:]))

    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)/look_back):
            a = dataset[look_back*i:look_back*(i+1),1:dim]
            dataX.append(a)
            dataY.append(dataset[look_back*i:look_back*(i+1), dim])
        return np.array(dataX), np.array(dataY)

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape for LSTM  -> (size, timestep, dim) = (150, year, 99)
    trainX_t, trainY_t = [],[]
    testX_t, testY_t = [],[]

    for j in range(seq-1):
        temp_t = trainX[j].reshape(area, 1, dim - 1)
        trainX_t.append(temp_t)
        target_t = trainY[j].reshape(area, 1)
        trainY_t.append(target_t)

    testX_t = testX.reshape(area, 1, dim-1)
    testY_t = testY.reshape(area, 1)

    # build LSTM from keras
    from keras.layers import LSTM
    from keras.models import Sequential
    from keras.layers import Dense
    import keras.backend as K

    K.clear_session()
    model = Sequential()  # Sequeatial Model
    model.add(LSTM(hidden_size, input_shape=(1, dim - 1), activation='hard_sigmoid'))  # (timestep, feature)
    model.add(Dense(1))  # output = 1
    RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()
    for k in range(seq - 1):
        print(k)
        model.fit(trainX_t[k], trainY_t[k], epochs=200, verbose=2, batch_size=84)

    # training
    #trainPredict = model.predict(trainX_t)
    testPredict = model.predict(testX_t)

    # testing
    #trainScore = model.evaluate(trainX_t, trainY_t)
    testScore = model.evaluate(testX_t, testY_t)
    #trainRMSE = math.sqrt(trainScore[0])
    testRMSE = math.sqrt(testScore[0])
    testPRMSE = math.sqrt(testScore[0]) * 100 / np.mean(testY)

    #print("train_rmse: ", trainRMSE)
    print("Year:" + str(real_year) + ", test_rmse: ", testRMSE)
    print("Year:" + str(real_year) + ", test_%rmse: ", testPRMSE)

    temp = [real_year, testRMSE, testPRMSE]
    temp = np.array(temp)
    temp = temp.reshape(1, 3)
    Result11 = np.append(Result11, temp)

Result11 = Result11.reshape(14, 3)
print(Result11)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result11[:,2]))

#plt.figure()
#plt.plot(testPredict)
#plt.plot(testY_t)

#####################################################################

input = np.genfromtxt('/home/deeplearning2/PycharmProjects/LHJ/soybean/soy_jun_sep_16day_78_11x7.csv', delimiter=',')
#input = sorted(input, key=lambda input: input[0])
year = 14
Result12 = []
Result12 = np.array(Result12)

for yr in range(year):
    real_year = 2003 + yr
# train parameters
    timestep = seq = 14
    dim = len(input[1,:])-1
    train_size = int(len(input) * 13/14)
    test_size = int(len(input) * 1/14)
    hidden_size = 800
    look_back = area = 168

    # train / test split
    test = input[test_size*yr:test_size*(yr+1)]
    if yr == 0:
        train = input[test_size:len(input)]
    elif yr == 13:
        train = input[0:train_size]
    else:
        train = np.append(input[0:test_size*yr], input[test_size*(yr+1):len(input)])
        train = train.reshape(train_size,len(input[1,:]))

    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)/look_back):
            a = dataset[look_back*i:look_back*(i+1),1:dim]
            dataX.append(a)
            dataY.append(dataset[look_back*i:look_back*(i+1), dim])
        return np.array(dataX), np.array(dataY)

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape for LSTM  -> (size, timestep, dim) = (150, year, 99)
    trainX_t, trainY_t = [],[]
    testX_t, testY_t = [],[]

    for j in range(seq-1):
        temp_t = trainX[j].reshape(area, 1, dim - 1)
        trainX_t.append(temp_t)
        target_t = trainY[j].reshape(area, 1)
        trainY_t.append(target_t)

    testX_t = testX.reshape(area, 1, dim-1)
    testY_t = testY.reshape(area, 1)

    # build LSTM from keras
    from keras.layers import LSTM
    from keras.models import Sequential
    from keras.layers import Dense
    import keras.backend as K

    K.clear_session()
    model = Sequential()  # Sequeatial Model
    model.add(LSTM(hidden_size, input_shape=(1, dim - 1), activation='hard_sigmoid'))  # (timestep, feature)
    model.add(Dense(1))  # output = 1
    RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()
    for k in range(seq - 1):
        print(k)
        model.fit(trainX_t[k], trainY_t[k], epochs=200, verbose=2, batch_size=84)

    # training
    #trainPredict = model.predict(trainX_t)
    testPredict = model.predict(testX_t)

    # testing
    #trainScore = model.evaluate(trainX_t, trainY_t)
    testScore = model.evaluate(testX_t, testY_t)
    #trainRMSE = math.sqrt(trainScore[0])
    testRMSE = math.sqrt(testScore[0])
    testPRMSE = math.sqrt(testScore[0]) * 100 / np.mean(testY)

    #print("train_rmse: ", trainRMSE)
    print("Year:" + str(real_year) + ", test_rmse: ", testRMSE)
    print("Year:" + str(real_year) + ", test_%rmse: ", testPRMSE)

    temp = [real_year, testRMSE, testPRMSE]
    temp = np.array(temp)
    temp = temp.reshape(1, 3)
    Result12 = np.append(Result12, temp)

Result12 = Result12.reshape(14, 3)
print(Result12)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result12[:,2]))

#plt.figure()
#plt.plot(testPredict)
#plt.plot(testY_t)

#####################################################################

input = np.genfromtxt('/home/deeplearning2/PycharmProjects/LHJ/soybean/soy_may_aug_16day_78_11x7.csv', delimiter=',')
#input = sorted(input, key=lambda input: input[0])
year = 14
Result13 = []
Result13 = np.array(Result13)

for yr in range(year):
    real_year = 2003 + yr
# train parameters
    timestep = seq = 14
    dim = len(input[1,:])-1
    train_size = int(len(input) * 13/14)
    test_size = int(len(input) * 1/14)
    hidden_size = 800
    look_back = area = 168

    # train / test split
    test = input[test_size*yr:test_size*(yr+1)]
    if yr == 0:
        train = input[test_size:len(input)]
    elif yr == 13:
        train = input[0:train_size]
    else:
        train = np.append(input[0:test_size*yr], input[test_size*(yr+1):len(input)])
        train = train.reshape(train_size,len(input[1,:]))

    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)/look_back):
            a = dataset[look_back*i:look_back*(i+1),1:dim]
            dataX.append(a)
            dataY.append(dataset[look_back*i:look_back*(i+1), dim])
        return np.array(dataX), np.array(dataY)

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape for LSTM  -> (size, timestep, dim) = (150, year, 99)
    trainX_t, trainY_t = [],[]
    testX_t, testY_t = [],[]

    for j in range(seq-1):
        temp_t = trainX[j].reshape(area, 1, dim - 1)
        trainX_t.append(temp_t)
        target_t = trainY[j].reshape(area, 1)
        trainY_t.append(target_t)

    testX_t = testX.reshape(area, 1, dim-1)
    testY_t = testY.reshape(area, 1)

    # build LSTM from keras
    from keras.layers import LSTM
    from keras.models import Sequential
    from keras.layers import Dense
    from keras import optimizers
    import keras.backend as K

    K.clear_session()
    model = Sequential()  # Sequeatial Model
    model.add(LSTM(hidden_size, input_shape=(1, dim - 1), activation='hard_sigmoid'))  # (timestep, feature)
    model.add(Dense(1))  # output = 1
    RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()
    for k in range(seq - 1):
        print(k)
        model.fit(trainX_t[k], trainY_t[k], epochs=200, verbose=2, batch_size=84)

    # training
    #trainPredict = model.predict(trainX_t)
    testPredict = model.predict(testX_t)

    # testing
    #trainScore = model.evaluate(trainX_t, trainY_t)
    testScore = model.evaluate(testX_t, testY_t)
    #trainRMSE = math.sqrt(trainScore[0])
    testRMSE = math.sqrt(testScore[0])
    testPRMSE = math.sqrt(testScore[0]) * 100 / np.mean(testY)

    #print("train_rmse: ", trainRMSE)
    print("Year:" + str(real_year) + ", test_rmse: ", testRMSE)
    print("Year:" + str(real_year) + ", test_%rmse: ", testPRMSE)

    temp = [real_year, testRMSE, testPRMSE]
    temp = np.array(temp)
    temp = temp.reshape(1, 3)
    Result13 = np.append(Result13, temp)

Result13 = Result13.reshape(14, 3)
print(Result13)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result13[:,2]))

#plt.figure()
#plt.plot(testPredict)
#plt.plot(testY_t)

#####################################################################

input = np.genfromtxt('/home/deeplearning2/PycharmProjects/LHJ/soybean/soy_may_sep_16day_100_11x9.csv', delimiter=',')
#input = sorted(input, key=lambda input: input[0])
year = 14
Result14 = []
Result14 = np.array(Result14)

for yr in range(year):
    real_year = 2003 + yr
# train parameters
    timestep = seq = 14
    dim = len(input[1,:])-1
    train_size = int(len(input) * 13/14)
    test_size = int(len(input) * 1/14)
    hidden_size = 800
    look_back = area = 168

    # train / test split
    test = input[test_size*yr:test_size*(yr+1)]
    if yr == 0:
        train = input[test_size:len(input)]
    elif yr == 13:
        train = input[0:train_size]
    else:
        train = np.append(input[0:test_size*yr], input[test_size*(yr+1):len(input)])
        train = train.reshape(train_size,len(input[1,:]))

    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)/look_back):
            a = dataset[look_back*i:look_back*(i+1),1:dim]
            dataX.append(a)
            dataY.append(dataset[look_back*i:look_back*(i+1), dim])
        return np.array(dataX), np.array(dataY)

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape for LSTM  -> (size, timestep, dim) = (150, year, 99)
    trainX_t, trainY_t = [],[]
    testX_t, testY_t = [],[]

    for j in range(seq-1):
        temp_t = trainX[j].reshape(area, 1, dim - 1)
        trainX_t.append(temp_t)
        target_t = trainY[j].reshape(area, 1)
        trainY_t.append(target_t)

    testX_t = testX.reshape(area, 1, dim-1)
    testY_t = testY.reshape(area, 1)

    # build LSTM from keras
    from keras.layers import LSTM
    from keras.models import Sequential
    from keras.layers import Dense
    import keras.backend as K

    K.clear_session()
    model = Sequential()  # Sequeatial Model
    model.add(LSTM(hidden_size, input_shape=(1, dim - 1), activation='hard_sigmoid'))  # (timestep, feature)
    model.add(Dense(1))  # output = 1
    RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()
    for k in range(seq - 1):
        print(k)
        model.fit(trainX_t[k], trainY_t[k], epochs=200, verbose=2, batch_size=84)

    # training
    #trainPredict = model.predict(trainX_t)
    testPredict = model.predict(testX_t)

    # testing
    #trainScore = model.evaluate(trainX_t, trainY_t)
    testScore = model.evaluate(testX_t, testY_t)
    #trainRMSE = math.sqrt(trainScore[0])
    testRMSE = math.sqrt(testScore[0])
    testPRMSE = math.sqrt(testScore[0]) * 100 / np.mean(testY)

    #print("train_rmse: ", trainRMSE)
    print("Year:" + str(real_year) + ", test_rmse: ", testRMSE)
    print("Year:" + str(real_year) + ", test_%rmse: ", testPRMSE)

    temp = [real_year, testRMSE, testPRMSE]
    temp = np.array(temp)
    temp = temp.reshape(1, 3)
    Result14 = np.append(Result14, temp)

Result14 = Result14.reshape(14, 3)
print(Result14)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result14[:,2]))

#plt.figure()
#plt.plot(testPredict)
#plt.plot(testY_t)

#####################################################################

print("---------------------RICE----------------------")
print("%RMSE of rice yield prediction (4~8)")
print(Result15)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result15[:,2]))
print("---------------------------------------------------")
print("%RMSE of rice yield prediction (4~9)")
print(Result16)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result16[:,2]))
print("---------------------------------------------------")
print("%RMSE of rice yield prediction (6~8)")
print(Result17)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result17[:,2]))
print("---------------------------------------------------")
print("%RMSE of rice yield prediction (6~9)")
print(Result18)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result18[:,2]))
print("---------------------------------------------------")
print("%RMSE of rice yield prediction (5~8)")
print(Result19)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result19[:,2]))
print("---------------------------------------------------")
print("%RMSE of rice yield prediction (5~9)")
print(Result20)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result20[:,2]))
print("---------------------------------------------------")

#####################################################################

print("----------------------CORN-----------------------")
print("%RMSE of corn yield prediction (4~8)")
print(Result3)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result3[:,2]))
print("---------------------------------------------------")
print("%RMSE of corn yield prediction (4~9)")
print(Result4)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result4[:,2]))
print("---------------------------------------------------")
print("%RMSE of corn yield prediction (6~8)")
print(Result5)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result5[:,2]))
print("---------------------------------------------------")
print("%RMSE of corn yield prediction (6~9)")
print(Result6)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result6[:,2]))
print("---------------------------------------------------")
print("%RMSE of corn yield prediction (5~8)")
print(Result7)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result7[:,2]))
print("---------------------------------------------------")
print("%RMSE of corn yield prediction (5~9)")
print(Result8)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result8[:,2]))
print("---------------------------------------------------")

#####################################################################

print("------------------SOYBEAN--------------------")
print("%RMSE of soybean yield prediction (4~8)")
print(Result9)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result9[:,2]))
print("---------------------------------------------------")
print("%RMSE of soybean yield prediction (4~9)")
print(Result10)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result10[:,2]))
print("---------------------------------------------------")
print("%RMSE of soybean yield prediction (6~8)")
print(Result11)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result11[:,2]))
print("---------------------------------------------------")
print("%RMSE of soybean yield prediction (6~9)")
print(Result12)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result12[:,2]))
print("---------------------------------------------------")
print("%RMSE of soybean yield prediction (5~8)")
print(Result13)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result13[:,2]))
print("---------------------------------------------------")
print("%RMSE of soybean yield prediction (5~9)")
print(Result14)
print("Average %RMSE of 2003 ~ 2016: ", np.mean(Result14[:,2]))
print("---------------------------------------------------")
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime as dt
from pylab import rcParams
mpl.style.use('ggplot')
from forecasting import *
from geneticModel import *
from misc import *

filepath = "../data/PJM_Jan2012_Dec2015.mat"  # SET FILE PATH TO DATA

# LOAD DATA SET

data = loadData('MISO', 'rt', 'lmp')
data.dropna(axis=1,how='any', inplace=True)
data.fillna(np.mean(data, axis=0), inplace=True)
columns = data.columns
data.columns = range(data.shape[1])

# PARAMETERS

hoursBack = list(np.arange(1, 11, 1))
trainShift = 50
numberOfWindows = 10


class TestParam(object):
    node = 2
    eta = 0.7  # exponential weights parameter
    alpha = 0.2  # sharing parameter - minimum weight of each model = alpha / number of Models
    startTimeStep = 10000
    # prediction range length same as training length
    endTimeStep = startTimeStep + 2000
    offset = 24
    lossFun = absError
    numPred = 1
    hoursBack = hoursBack


# TRANSFORM DATA

dataTrans = pd.DataFrame(dataTransform(data, 50, 13000, hoursBack=hoursBack))


# TRAINING MODEL

time0 = dt.datetime.now()
print('Running...')
# Run this for a number of windows
experts_errorRED = []
partitionsRED = []
expertsRED = []
for i in range(numberOfWindows):
    error, partition, expert = geneticSelection(data, dataTrans,
                                                         numModels=150,
                                                         numParts=10,
                                                         numOffspring=2,
                                                         numSwap=5,
                                                         alpha=1,
                                                         startTimeStep=200 + i * trainShift,
                                                         trainingWindow=1000,
                                                         threshold=1.05,
                                                         offset=24,
                                                         numGenerations=1,
                                                         expertsReturn=1,
                                                         hoursBack=hoursBack,
                                                         lossFun=RMSE)

    experts_errorRED += error
    partitionsRED += partition
    expertsRED += expert

    print("_________________________________Training window {} complete________________________________".format(i + 1))
    time1 = dt.datetime.now()
    print('total:', (time1-time0).seconds, 'sec')

time1 = dt.datetime.now()
print('total:', (time1-time0).seconds, 'sec')


# TEST MODEL


time0 = dt.datetime.now()


weights, forecast, modelForecast = fixedShareAlg_varUpdate_nodeWeights(data, dataTrans, expertsRED, partitionsRED, TestParam.eta,
                                                           TestParam.alpha,
                                                           TestParam.startTimeStep, TestParam.endTimeStep,
                                                           TestParam.offset, RMSE_weightNodes, TestParam.numPred,
                                                           TestParam.hoursBack)

elapsed_time =  dt.datetime.now() - time1
print(elapsed_time)


cumError = cumSumError(data, forecast)

prev24 = data.ix[forecast.index[0] - 24: forecast.index[-1] - 24,:]
prev24.index = forecast.index
prev24 = cumSumError(data, prev24)


rcParams['figure.figsize'] = 15, 5

plt.plot(cumError)
plt.plot(prev24)
plt.xlabel('Hour')
plt.ylabel('Cumlative RMSE')
plt.title('Cumulative Error of forecast DA LMP')
plt.legend(['Cumulative Error', 'Previous 24 hours'])


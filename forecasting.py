import numpy as np
import pandas as pd


def get_testing_data(dataTrans, nodes, startTimeStep, endTimeStep, hoursBack=0):
    """
    Function to get testing data for current expert in partition

    Parameters
    ----------
    data: Pandas DataFrame
        Data set we want run the algorithm on
    dataTrans: Pandas DataFrame
        Transformed data set used to get generate pattern target pairs
    nodes: list
        which current expert predicts on
    startTimeStep: int
        Start time of when we want to run the algorithm on the data. This should be a value within the index on the
        data.
    endTimeStep: int
        Number of data points we want to predict in the future
    hoursBack: int or list
        The previous hours we want to concatenate in the data

    Returns
    ----------
    X: numpy array
        pattern
    Y: numpy array
        target

    """
    # select correct nodes from hours back
    if isinstance(hoursBack, list):
        nodesBack = np.arange(len([0] + hoursBack))
        origDataNodes = dataTrans.shape[1] / len(nodesBack)
        nodes = [node + origDataNodes * nb for node in nodes for nb in nodesBack]
    elif not hoursBack == 0:
        nodesBack = np.arange(len([0] + np.arange(hoursBack)))
        origDataNodes = dataTrans.shape[1] / len(nodesBack)
        nodes = [node + origDataNodes * nb for node in nodes for nb in nodesBack]

    X = np.array(dataTrans.iloc[startTimeStep: startTimeStep + endTimeStep, nodes])

    return X


def fixedShareAlg_varUpdate_nodeWeights(data, dataTrans, experts, expertNodes, eta, alpha, startTimeStep, endTimeStep, \
                                        offset, lossFun, numPred=1, hoursBack=0):
    """
    Fixed Share Forecasting Algorithm where forecast and weights are updated after the offset period, difference between
    X and Y. Here we are predicting all nodes at the same time.

    Parameters
    ----------
    data: Pandas DataFrame
        Dataset we want run the algorithm on
    dataTrans: Pandas DataFrame
        Transformed dataset used to get generate pattern target pairs
    experts: List of list of sklearn model
        Each sublist of sklearn models will be represent a partition
    expertNodes: List of list
        Each sublist represents a partition, corresponding to an expert in experts
    eta: float in [0, 1]
        Fixed share algorithm parameter
    alpha: float in [0, 1]
        Fixed share algorithm parameter
    startTimeStep: int
        Start time of when we want to run the algorithm on the data. This should be a value within the index on the
        data.
    endTimeStep: int
        When to stop running the algorithm. This should be a value within the index on the
        data.
    offset: int
        Difference between the pattern and target pairs
    lossFun: func
        Loss function to be used in the algorithm
    numPred: int
        Number of datapoints we want to predict in the future
    hoursBack: int or list
        The previous hours we want to concatenate in the data

    Return
    ----------

    weights: Pandas DataFrame
        Weights of all models for all time steps the algorithm ran on
    forecast: Pandas DataFrame
        Final forescsat
    modelForecast: Pandas DataFrame
        Forecast each model

    """

    # numPred, number of prediction in hte future we want to make before making a weight update
    numNodes = data.shape[1]  # number of node each expert predicts on
    numExperts = len(experts)  # number of experts
    timeSteps = endTimeStep - startTimeStep + numPred  # overall number of time steps
    weights = np.ones([timeSteps + 1, numExperts, numNodes]) * (1 / numExperts)  # initialising weights
    forecast = np.zeros([timeSteps, numNodes])  # array to store forecast
    modelForecast = np.ones([timeSteps, numExperts, numNodes])  # each time step has prediction fo each node expert
    # trained on, this is why we need to add third dimension to array
    # at each time step for each expert we store the prediction for all nodes that that expert predicts on.
    nextObs = np.zeros([numPred, numNodes])

    if timeSteps % numPred != 0:
        raise ValueError('Number of time steps need to be divisible by the number of predictions')

    for step in range(0, timeSteps, numPred):  # for each observation -
        # we are considering the n+1 observation at each time step
        weightsHat = np.zeros((numExperts, numNodes))

        if np.mod(step, 11) == 0:
            print(step)

        # make forecast for the next 24 hours - check error with those then update weights
        for expertIndex, expert in enumerate(experts):  # for each partition
            for subExpertIndex, subExpert in enumerate(expert):  # for each subset in partition

                prevObs = get_testing_data(dataTrans, expertNodes[expertIndex][subExpertIndex],
                                           startTimeStep, numPred, hoursBack)
                # modelPredictions for each expert, append to array for each of the next 24 hours
                # also store current observation for each expert
                modelForecast[step: step + numPred, expertIndex, expertNodes[expertIndex][subExpertIndex]] = \
                    subExpert.predict(prevObs)

            nextObs = data.iloc[step + startTimeStep + offset: step + startTimeStep + offset + numPred, :].values

            # Update the weights using exponential and loss function
            weightsHat[expertIndex, :] = weights[step, expertIndex, :] * \
                                         np.exp(- eta * lossFun(modelForecast[step: step + offset, expertIndex, :],
                                                                nextObs))

        # forecast is now updated back to original place, now we make forecast before we see/update the weights
        forecast[step: step + numPred, :] = np.sum(modelForecast[step: step + numPred, :, :] *
                                                   weights[step, :, :].reshape(1, numExperts, numNodes), axis=1)

        # repeat the array so that dimension are consistent
        weightsHat = weightsHat / np.array([np.sum(weightsHat, axis=0), ] * numExperts)
        weights[step + 1, :, :] = (1-alpha) * weightsHat + alpha * 1/numExperts

    # move data into pandas data frame for output assign the index equal to the index of the corresponding data
    # predictions

    forecast = pd.DataFrame(forecast)
    forecast.index = data.iloc[startTimeStep + offset: endTimeStep + offset + numPred, :].index
    try:
        modelForecast = pd.DataFrame(modelForecast)
        modelForecast.index = forecast.index
    except ValueError:
        pass
    finally:
        print("Model Forecast cannot be converted into a dataframe")

    return weights, forecast, modelForecast


def RMSE_weightNodes(x, y):
    return np.mean(np.sqrt((x - y) ** 2), axis=0)


def RMSE(x, y):
    return np.mean(np.sqrt((x - y) ** 2))







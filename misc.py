from scipy import io
from forecasting import RMSE_1
import numpy as np
import pandas as pd

def loadData(filepath, market='PJM', variable='da', comp='lmp'):
    """
    Function to load market data

    Param
    --------
    filepath: string
        file path of data
    market: string
        The market of data we want
    variable: string
        Day ahead (da) or real time (rt)
    comp: string
        Composition of data LMP (lmp), MCC (mcc), MEP (mep), MLP (mlp)

    Return
    --------
    data: Pandas DataFrame
        Corresponding market data

    """
    if market == 'PJM':
        data = io.loadmat(filepath, variable_names=[variable + '_' + comp])
    else:
        data = io.loadmat(filepath, variable_names=[variable + '_' + comp])

    data = pd.DataFrame(data[variable + '_' + comp])

    return data


def cumSumError(data, forecast):
    """
    Function to calculate cumulative error

    Param
    --------
    data: Pandas DataFrame
        data we used
    forecast: Pandas DataFrame
        The forecast we want to calculate error for.


    Return
    --------
    data: Pandas DataFrame
        Corresponding market data
    """

    return np.cumsum(RMSE_1(forecast.values, data.ix[forecast.index[0]: forecast.index[-1], :].values))
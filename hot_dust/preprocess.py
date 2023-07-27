import xarray as xr
import hvplot.xarray 
import numpy as np 
#import matplotlib.pyplot as plt

def prepare_training_data():
    ds = xr.open_dataset('data/rt_nn_irdust_training_data.nc')
    return ds


def feature_histogram(variable):
    plt = variable.hvplot.hist()
    return plt  

def heat_map(variable): 
    plt.imshow(variable) 
    plt.show()  


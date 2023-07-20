import xarray as xr
import hvplot.xarray 
import numpy as np

def prepare_training_data():
    ds = xr.open_dataset('data/rt_nn_irdust_training_data.nc')
    return ds


def feature_histogram(variable):
    plt = variable.hvplot.hist()
    return plt 

def modified_histogram(variable): # See if the bins will make the data easier to read 
    n_bins = int(np.ceil(np.sqrt(len(variable))))  
    plt = plt = variable.hvplot.hist(bins=n_bins)
    return plt


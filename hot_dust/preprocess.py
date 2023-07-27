import xarray as xr
import hvplot.xarray 
import numpy as np 
import matplotlib.pyplot as plt

def prepare_training_data():
    ds = xr.open_dataset('data/rt_nn_irdust_training_data.nc')
    return ds


def feature_histogram(variable):
    plt = variable.hvplot.hist()
    return plt  

def heat_map(variable): 
    plt.imshow(variable) 
    plt.show()  

def split_training_data():  
    ds = xr.open_dataset('data/rt_nn_irdust_training_data.nc')
    rng = np.random.default_rng(1234)  
    rng.shuffle(ds)  

    # Define the percentages of the data to take
    train_ratio = 0.6 
    validate_ratio = 0.1 
    test_ratio = 0.3  

    # Get the amount of data to pull into the new dataset
    ds_len = len(ds) 
    train_len = train_ratio * ds_len 
    validate_len = validate_ratio * ds_len 
    test_len = test_ratio * ds_len   

    # Split into train, validate, and test subsets 
    train_ds = ds[:train_len] # type: ignore
    validate_ds = ds[:validate_len] # type: ignore
    test_ds = ds[:test_len]  # type: ignore

    return test_ds






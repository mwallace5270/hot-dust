import xarray as xr
import hvplot.xarray 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 


def prepare_training_data():
    ds = xr.open_dataset('data/rt_nn_irdust_training_data.nc')
    return ds


def feature_histogram(variable):
    plt = variable.hvplot.hist()
    return plt  

def heat_map(variable): 
    plt.imshow(variable) 
    plt.colorbar()
    plt.show()  

def split_training_data():
    ds = xr.open_dataset('data/rt_nn_irdust_training_data.nc')   
        
    # Convert xarray to numpy array for shuffling   
    ds_np = ds.to_array().values

    # Random but reproducible data shuffle
    rng = np.random.default_rng(seed=1234)    
    ds_random = rng.permutation(ds_np) # .permutation instead of .shuffle 

    # Reshape to remove the extra dimension
    ds_random = ds_random.reshape(-1, ds_random.shape[-1]) 

    # Define the percentages of the data to take
    train_ratio = 0.6 
    validate_ratio = 0.1 
    test_ratio = 0.3  

    # Get the amount of data to pull into the new dataset
    ds_len = len(ds_random)  
    train_len = int(train_ratio * ds_len)
    validate_len = int(validate_ratio * ds_len)
    test_len = int(test_ratio * ds_len) 

    # Split into train, validate, and test subsets 
    X_train = ds_random[:train_len, :-1]
    y_train = ds_random[:train_len, -1]

    X_validate = ds_random[train_len:train_len + validate_len, :-1]
    y_validate = ds_random[train_len:train_len + validate_len, -1]

    X_test = ds_random[train_len + validate_len: train_len + validate_len + test_len, :-1]
    y_test = ds_random[train_len + validate_len: train_len + validate_len + test_len, -1]

    # Return the data splits as tf.data.Dataset constructed from tf.data.Dataset.from_tensor_slices 
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) 
    validate_dataset = tf.data.Dataset.from_tensor_slices((X_validate, y_validate))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    
    return X_test, y_test
    #return train_dataset, validate_dataset, test_dataset


    








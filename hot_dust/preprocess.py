import xarray as xr
import hvplot.xarray

def prepare_training_data():
    ds = xr.open_dataset('data/rt_nn_irdust_training_data.nc')
    return ds


def feature_histogram(variable):
    plt = variable.hvplot.hist()
    return plt
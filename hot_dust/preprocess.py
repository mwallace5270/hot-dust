from datetime import datetime

import hvplot.xarray
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def prepare_training_data():
    # Read the training data file
    ds = xr.open_dataset("data/rt_nn_irdust_training_data.nc")

    # Flatten multi-dimensional features (only viirs_bts)
    viirs_bts_labels = [f"bt_{int(i)}" for i in ds["nbands_viirs"].data]
    ds["nbands_viirs"] = viirs_bts_labels
    viirs_bts = ds["viirs_bts"].to_dataset(dim="nbands_viirs")
    ds = xr.merge((ds, viirs_bts))

    # Combine predictors into "x"
    features = [
        # "solar_zenith_angle",
        "viewing_zenith_angle",
        # "relative_azimuth_angle",
        "spress",  # surface pressure
        "h2o",  # water vapor
        "o3",  # ozone
        "ws",  # wind speed
        "ts",  # temperature
        *viirs_bts_labels,
    ]
    ds["x"] = xr.concat(
        [ds[i] for i in features],
        dim=xr.IndexVariable("features", features),
    )
    ds["x"].attrs = {}

    # Assign response (as the log of dust optical thickness) to "y"
    ds["y"] = np.log10(ds["dust_optical_thickness"])
    ds["y"].attrs = {}

    # Return just "x" and "y" with "sample" as the first dimension
    return ds[["x", "y"]].rename({"npoints": "sample"}).transpose("sample", ...)


def feature_histogram(ds, feature_name, feature_label):
    ds_sorted = ds.sortby("features")
    selected_feature = ds_sorted["x"].sel({"features": feature_name})
    plt = selected_feature.hvplot.hist(xlabel=feature_label, ylabel="Frequency", yformatter="%.0e") 
    plt = plt.opts(title="", fontscale=4, frame_height=500, frame_width=700)
    return plt


def feature_hexbin(ds, x_feature_name, y_feature_name):
    selected_feature = ds["x"].sel({"features": [x_feature_name, y_feature_name]})
    selected_feature = selected_feature.to_dataset("features")
    plt = selected_feature.hvplot.hexbin(x=x_feature_name, y=y_feature_name, aspect=1)
    return plt


def response_hexbin(ds, feature_name, response="y"):
    selected_feature = ds.sel({"features": feature_name})
    plt = selected_feature.hvplot.hexbin(x="x", y=response, aspect=1)
    return plt


def split_training_data(ds: xr.Dataset):
    # Assign a split variable with a random but reproducible category
    rng = np.random.default_rng(seed=3857382908474)
    ds["split"] = (
        # dimensions
        "sample",
        # values
        rng.choice(
            ["train", "validate", "test"],
            size=ds.sizes["sample"],
            p=[0.8, 0.15, 0.05],
        ),
    )

    # Split into train, validate, and test subsets
    ds = {key: value.drop_vars("split") for key, value in ds.groupby("split")}

    # Return the three datasets
    return ds["train"], ds["validate"], ds["test"]


def get_merra_variables(path: "pathlib.Path", fs: "s3fs.core.S3FileSystem" = None) -> xr.Dataset:
    # README
    # for opendap authentication, ensure correctness of ~/.netrc and ~/.dodsrc files
    # load attributes for temporal and spatial extents
    viirs = xr.open_dataset(path)
    tspan = [
        datetime.fromisoformat(viirs.attrs["time_coverage_start"].strip("Z")),
        datetime.fromisoformat(viirs.attrs["time_coverage_end"].strip("Z")),
    ]
    bbox = [
        viirs.attrs["geospatial_lon_min"],
        viirs.attrs["geospatial_lat_min"],
        viirs.attrs["geospatial_lon_max"],
        viirs.attrs["geospatial_lat_max"],
    ]
    # use day from tspan (assume no boundary crossing) to select MERRA-2 file
    ymd = tspan[0].strftime("%Y%m%d")
    granule = f"{ymd[:4]}/{ymd[4:6]}/MERRA2_400.inst1_2d_asm_Nx.{ymd}.nc4" 
    if fs:
        # earthdata cloud
        url = fs.open(f"s3://gesdisc-cumulus-prod-protected/MERRA2/M2I1NXASM.5.12.4/{granule}")
    else:
        # opendap
        url = f"dap4://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2I1NXASM.5.12.4/{granule}"        
    # select variables
    merra = xr.open_dataset(url, chunks={})
    ws = ["U10M", "V10M"] # 10-meter east|northward wind (10 and 50 meter also available)
    variable = [
        "PS",  # surface_pressure
        "TS",  # surface_skin_temperature
        "TO3",  # total_column_ozone
        "TQV",  # total_precipitable_water_vapor
        *ws,
    ]
    merra = merra[variable]
    # create wind speed
    merra["WS"] = (merra[ws].to_array(dim="vec") ** 2).sum(dim="vec") ** 0.5
    merra["WS"].attrs["long_name"] = "magnitude of 10-meter wind vector"
    merra = merra.drop_vars(ws)
    # identify coordinates for temporal and spatial interpolation
    tmean = np.array(tspan[0] + (tspan[1] - tspan[0]) / 2, dtype=merra["time"].dtype)
    points = xr.open_dataset(path, group="geolocation_data")
    interp = merra.interp(
        coords={
            "time": tmean,
            "lon": points["longitude"],
            "lat": points["latitude"],
        },
        method="linear",
    )
    return interp


def process_granule(path: "pathlib.Path") -> xr.Dataset:
    # ## File paths
    stem = path.stem.split(".")
    granule_id = ".".join(stem[1:3])
    vnp02 = next(path.parent.glob(f"VNP02MOD.{granule_id}.*"))
    vnp03 = next(path.parent.glob(f"VNP03MOD.{granule_id}.*"))
    clmsk = next(path.parent.glob(f"CLDMSK_L2_VIIRS_SNPP.{granule_id}.*"))
    merra = path.parents[1] / "merra" / vnp03.name
    
    # ## Load VNP02MOD level-1 data
    # open WITHOUT mask_and_scale to access observations as integers (ignore scale and offset)
    open_vnp02_int = xr.open_dataset(vnp02, group="observation_data", mask_and_scale=False)
    # open WITH mask_and_scale to access LUT values as floats (with NaNs)
    open_vnp02 = xr.open_dataset(vnp02, group="observation_data")
    # create new xr.Dataset to contain viirs_bts
    vnp02_variables = xr.Dataset()
    for item in ["M14", "M15", "M16"]:
        vnp02_variables[item + "_bt"] = open_vnp02[item + "_brightness_temperature_lut"][open_vnp02_int[item]]
        long_name = vnp02_variables[f"{item}_bt"].attrs["long_name"]
        vnp02_variables[item + "_bt"].attrs["long_name"] = long_name.replace(" lookup table", "")
    
    # ## Load VNP03MOD geolocation coordinates
    # open (just open, no tricks here)
    open_vnp03 = xr.open_dataset(vnp03, group="geolocation_data")    
    # select the relavant variables
    vnp03_variables = open_vnp03[["sensor_zenith", "solar_zenith"]]
    
    # ## Load CLDMSK_L2_VIIRS_SNPP cloud mask
    # keep "probably clear" and "confident clear"
    open_clmsk = xr.open_dataset(clmsk, group="geophysical_data", mask_and_scale=False)
    cloud_mask = open_clmsk["Integer_Cloud_Mask"] >= 2
    land_flag_mask = np.array(0b1100_0000, dtype=np.dtype("B"))
    land_flag_value = np.array(0b0000_0000, dtype=np.dtype("B"))
    land_mask = np.bitwise_and(open_clmsk["Cloud_Mask"][0], land_flag_mask) == land_flag_value    
    vnp02_variables = vnp02_variables.where(land_mask)  # (cloud_mask & land_mask)
    
    # ## Load MERRA data
    open_merra = xr.open_dataset(merra)    
    # drop the coordinates 
    merra_variables = open_merra.drop(['time', 'lon', 'lat'])
    
    # ## Process and Merge Inputs
    # compute relative azimuth
    array = np.abs(open_vnp03["solar_azimuth"] - open_vnp03["sensor_azimuth"])
    vnp03_variables["relative_azimuth"] = array.where(array <= 180, 360 - array)
    # divide the MERRA2 pressure by 100 to get it in the right units
    merra_variables['PS'] = merra_variables['PS'] / 100
    # merge viirs and merra variables
    variables_merged = xr.merge([vnp02_variables, vnp03_variables, merra_variables])
    
    # ## Transform to Model Input
    # the features list must references the same variables as the features list in `prepare_training_data`
    features = [
        # "solar_zenith",
        "sensor_zenith",
        # "relative_azimuth",
        "PS",
        "TQV",
        "TO3",
        "WS",
        "TS",
        "M14_bt",
        "M15_bt",
        "M16_bt",
    ]    
    # concatenate the feature variables, in the order above, into a 3D array
    x = variables_merged[features].to_array("feature")
    
    return x 

def sensitivity_analysis(ds, network, percentage):
    x_values = ds['x'].values
    sensitivity_values = [] 
    std_dev = np.std(x_values)    

    perturbation = 0.01 * std_dev * percentage  # pertubation is 1% of standard deviation
    # Peturb the x values
    perturbed_values = x_values + perturbation # x + dx 
    original_outputs = network.predict(x_values)
    
    #loop for each variable 
    for i in range( x_values.shape[1]):
        x = x_values.copy() 
        x[:,i] = perturbed_values[:,i]
        # Predict with perturbed values
        perturbed_outputs = network.predict(x)
        # Calculate sensitivity: (change in y) / (change in x) 
        sensitivity = np.mean((perturbed_outputs - original_outputs) / (perturbation - x_values))
        sensitivity_values.append(sensitivity) 

    # Plot the sensitivity matrix and histogram 
    plt.plot(sensitivity_values)
    plt.xlabel('Percentage of Pertubation')
    plt.ylabel("Sensitivity") 
    plt.title("Sensitivity Analysis" )
    plt.show() 
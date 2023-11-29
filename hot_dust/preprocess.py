from datetime import datetime

import hvplot.xarray
import numpy as np
import xarray as xr
from scipy.interpolate import interpn

from . import DATADIR


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


def feature_histogram(ds, feature_name):
    selected_feature = ds["x"].sel({"features": feature_name})
    plt = selected_feature.hvplot.hist()
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


def get_merra_variables(granule: str) -> xr.Dataset:
    # README
    # for opendap authentication, ensure correctness of ~/.netrc and ~/.dodsrc files
    # load attributes for temporal and spatial extents
    viirs = xr.open_dataset(DATADIR / "granules" / granule)
    tspan = [
        datetime.fromisoformat(viirs.attrs["time_coverage_start"]),
        datetime.fromisoformat(viirs.attrs["time_coverage_end"]),
    ]
    bbox = [
        viirs.attrs["geospatial_lon_min"],
        viirs.attrs["geospatial_lat_min"],
        viirs.attrs["geospatial_lon_max"],
        viirs.attrs["geospatial_lat_max"],
    ]
    # use day from tspan (assume no boundary crossing) to select MERRA-2 file
    ymd = tspan[0].strftime("%Y%m%d")
    url = "dap4://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2I1NXASM.5.12.4/{}"
    path = f"{ymd[:4]}/{ymd[4:6]}/MERRA2_400.inst1_2d_asm_Nx.{ymd}.nc4"
    merra = xr.open_dataset(url.format(path))
    # select variables
    variable = [
        "PS",  # surface_pressure
        "TS",  # surface_skin_temperature
        "TO3",  # total_column_ozone
        "TQV",  # total_precipitable_water_vapor
    ]
    merra = merra[variable]
    # identify coordinates for temporal and spatial interpolation
    tmean = np.datetime64(tspan[0] + (tspan[1] - tspan[0]) / 2)
    points = xr.open_dataset(DATADIR / "granules" / granule, group="geolocation_data")
    interp = merra.interp(
        coords={
            "time": tmean,
            "lon": points["longitude"],
            "lat": points["latitude"],
        },
        method="linear",
    )
    return interp

import xarray as xr
import hvplot.xarray
import numpy as np


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
        "solar_zenith_angle",
        "viewing_zenith_angle",
        "relative_azimuth_angle",
        "spress",
        "h2o",
        "o3",
        "ws",
        "ts",
        *viirs_bts_labels,
    ]
    ds["x"] = xr.concat(
        [ds[i] for i in features],
        dim=xr.IndexVariable("features", features),
    )
    ds["x"].attrs = {}

    # Assign response (as the log of dust optical thickness) to "y"
    ds["y"] = np.log10(ds["dust_optical_thickness"])  # log10
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

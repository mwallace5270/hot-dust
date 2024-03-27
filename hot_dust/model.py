import json

import tensorflow as tf
import xarray as xr


def pretraining(dataset):
    # create a non-trainable layer to normalize the features
    layer = tf.keras.layers.Normalization()
    layer.adapt(dataset.map(lambda x, y: x).batch(2**12))
    return layer


def compile(normalization, **kwargs):
    ## build model
    # single input layer
    inputs = tf.keras.Input(shape=normalization.weights[0].shape)
    layer = normalization(inputs)
    # hidden layers
    layer = tf.keras.layers.Dense(16, "relu")(layer)
    layer = tf.keras.layers.Dense(16, "relu")(layer)
    # prediction layer
    output = tf.keras.layers.Dense(2, "linear")(layer) 
    # create network
    network = tf.keras.Model(inputs=inputs, outputs=output)
    # add optimizer, loss, and any keyword arguments from call
    network.compile(
        optimizer=tf.optimizers.Adam(learning_rate=3e-4),
        loss=tf.keras.losses.MeanSquaredError(),
        **kwargs,
        # run_eagerly=True,  # DEBUG
    )
    return network


def save_metrics(network, dataset, path, **kwargs):
    ## add metrics for evaluation only
    network.compile(
        **kwargs,
        metrics=[
            # TODO
            # tf.keras.metrics.MeanMetricWrapper(
            #     fn=lambda y_true, y_pred: y_pred - y_true,
            #     name="ME",
            # ),
            # tf.keras.metrics.MeanAbsoluteError(name="MAE"),
            # tf.keras.metrics.RootMeanSquaredError(name="RMSE"),
            # tfa.metrics.RSquare(name="R2"),
        ],
    )
    ## calculate metrics and write to json
    metrics = {k.name: v for k, v in zip(network.metrics, network.evaluate(dataset))}
    with path.open("w") as stream:
        json.dump(metrics, stream)


def plot_loss(trace):
    ds = xr.Dataset(
        {
            "epoch": ("epoch", trace["epoch"]),
            "training": ("epoch", trace["loss"]),
            "validation": ("epoch", trace["val_loss"]),
        },
    )
    plot = ds.hvplot.scatter(logy=True, ylabel="MSE", group_label="split")
    plot.opts(legend_position="top_right", fontscale=4, frame_height=500, frame_width=700)
    return plot


def dummy_data():
    mnist = tf.keras.datasets.mnist
    (x, y), *_ = mnist.load_data()
    n = x.shape[0]
    x = x.reshape((n, -1))
    return (
        tf.data.Dataset.from_tensor_slices((x, y)),
        tf.data.Dataset.from_tensor_slices((x[: (n // 5), :], y[: (n // 5)])),
        tf.data.Dataset.from_tensor_slices((x[: (n // 5), :], y[: (n // 5)])),
    ) 


def to_tensorflow(ds):
    dataset = []
    for item in ds:
        dataset.append(
            tf.data.Dataset.from_tensor_slices((item["x"].values, item["y"].values))
        )
    return tuple(dataset)

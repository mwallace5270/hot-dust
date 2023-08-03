import json

import tensorflow as tf
import xarray as xr
import hvplot.xarray


def pretraining(dataset):
    # create a non-trainable layer normalize the
    layer = tf.keras.layers.Normalization()
    layer.adapt(dataset.batch(2**12).map(lambda x, y: x).take(1))
    return layer


def compile(normalization, **kwargs):
    ## build model
    # single input layer
    inputs = tf.keras.Input(shape=normalization.weights[0].shape)
    layer = normalization(inputs)
    # hidden layers
    # TODO 
    hidden1 = tf.keras.layers.Dense(128, 'relu')(layer) 
    hidden2 = tf.keras.layers.Dense(64, 'relu')(hidden1) 
    # prediction layer
    hidden = tf.keras.layers.Dense(64, "relu")(hidden1)
    outputs =  hidden = tf.keras.layers.Dense(1, "linear")(hidden)
    # create network
    network = tf.keras.Model(inputs=inputs, outputs=outputs)
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
            "train_loss": ("epoch", trace["loss"]),
            "validation_loss": ("epoch", trace["val_loss"]),
        },
    )
    return ds.hvplot.scatter()


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

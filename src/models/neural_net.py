"""Keras MLP regressor for Energy Consumed prediction."""

import tensorflow as tf
from tensorflow import keras


def build_neural_net(input_dim: int, cfg: dict) -> keras.Model:
    nn_cfg = cfg["models"]["neural_net"]

    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for units in nn_cfg["hidden_layers"]:
        x = keras.layers.Dense(units, activation="relu")(x)
        x = keras.layers.Dropout(nn_cfg["dropout"])(x)
    outputs = keras.layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=nn_cfg["learning_rate"]),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_neural_net(X_train, y_train, X_val, y_val, cfg: dict):
    nn_cfg = cfg["models"]["neural_net"]
    tf.random.set_seed(cfg["random_seed"])

    model = build_neural_net(input_dim=X_train.shape[1], cfg=cfg)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=nn_cfg["early_stopping_patience"],
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=nn_cfg["epochs"],
        batch_size=nn_cfg["batch_size"],
        callbacks=[early_stopping],
        verbose=0,
    )

    print(
        f"Neural net trained for {len(history.history['loss'])} epochs "
        f"(best val_loss {min(history.history['val_loss']):.3f})"
    )

    return model, history

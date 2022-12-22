import numpy as np, pandas as pd
import joblib
import sys
import os, warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
warnings.filterwarnings("ignore")


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.losses import (
    CategoricalCrossentropy,
    SparseCategoricalCrossentropy,
)


model_params_fname = "model_params.save"
model_wts_fname = "model_wts.save"
history_fname = "history.json"
MODEL_NAME = "multi_class_base_simple_ann_tfkeras"

COST_THRESHOLD = float("inf")


class InfCostStopCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        loss_val = logs.get("loss")
        if loss_val == COST_THRESHOLD or tf.math.is_nan(loss_val):
            print("Cost is inf, so stopping training!!")
            self.model.stop_training = True


class Classifier:
    def __init__(self, D, K, l1_reg=1e-3, l2_reg=1e-1, lr=1e-3, **kwargs) -> None:
        self.D = D
        self.K = K
        self.l1_reg = np.float(l1_reg)
        self.l2_reg = np.float(l2_reg)
        self.lr = lr

        self.net = self.build_model()
        self.net.compile(
            loss=SparseCategoricalCrossentropy(),
            # optimizer=Adam(learning_rate=self.lr),
            optimizer=SGD(learning_rate=self.lr),
            metrics=["accuracy"],
        )

    def build_model(self):
        reg = l1_l2(l1=self.l1_reg, l2=self.l2_reg)
        input_ = Input(self.D)
        x = input_
        x = Dense(self.K, activity_regularizer=reg, activation="softmax")(x)
        output_ = x
        model = Model(input_, output_)
        # model.summary(); sys.exit()
        return model

    def fit(
        self,
        train_X,
        train_y,
        valid_X=None,
        valid_y=None,
        batch_size=256,
        epochs=100,
        verbose=0,
    ):

        if valid_X is not None and valid_y is not None:
            early_stop_loss = "val_loss"
            validation_data = [valid_X, valid_y]
        else:
            early_stop_loss = "loss"
            validation_data = None

        early_stop_callback = EarlyStopping(
            monitor=early_stop_loss, min_delta=1e-3, patience=3
        )
        infcost_stop_callback = InfCostStopCallback()

        history = self.net.fit(
            x=train_X,
            y=train_y,
            batch_size=batch_size,
            validation_data=validation_data,
            epochs=epochs,
            verbose=verbose,
            shuffle=True,
            callbacks=[early_stop_callback, infcost_stop_callback],
        )
        return history

    def predict(self, X, verbose=False):
        preds = self.net.predict(X, verbose=verbose)
        return preds

    def summary(self):
        self.net.summary()

    def evaluate(self, x_test, y_test):
        """Evaluate the model and return the loss and metrics"""
        if self.net is not None:
            return self.net.evaluate(x_test, y_test, verbose=0)

    def save(self, model_path):
        model_params = {
            "D": self.D,
            "K": self.K,
            "l1_reg": self.l1_reg,
            "l2_reg": self.l2_reg,
            "lr": self.lr,
        }
        joblib.dump(model_params, os.path.join(model_path, model_params_fname))
        self.net.save_weights(os.path.join(model_path, model_wts_fname))

    @classmethod
    def load(cls, model_path):
        # print(model_params_fname, model_wts_fname)
        model_params = joblib.load(os.path.join(model_path, model_params_fname))
        classifier = cls(**model_params)
        classifier.net.load_weights(
            os.path.join(model_path, model_wts_fname)
        ).expect_partial()
        return classifier


def save_model(model, model_path):
    model.save(model_path)


def load_model(model_path):
    try:
        model = Classifier.load(model_path)
    except:
        raise Exception(
            f"""Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?"""
        )
    return model


def save_training_history(history, f_path):
    hist_df = pd.DataFrame(history.history)
    hist_json_file = os.path.join(f_path, history_fname)
    with open(hist_json_file, mode="w") as f:
        hist_df.to_json(f)


def get_data_based_model_params(X, y):
    """
    Set any model parameters that are data dependent.
    For example, number of layers or neurons in a neural network as a function of data shape.
    """
    return {"D": X.shape[1], "K": len(set(y))}

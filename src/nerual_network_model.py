from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import nadam, sgd
import numpy as np

import keras.backend as K

from sample_allocation import get_wine_data, WineData, QualityLabels, CHEM_ATTR_KEYS

"""
Cody Whitt
pkz325
CPSC 4240 Spring 2023
Final Project

ANN Model stuff. Currently very basic, but will be made dynamic in terms of pseudo-grid search / cross-validation
type stuff (similar to sample_allocation.py stuff).
"""


def train_test_ann(wine_data: WineData):
    """
    Doc Doc Doc
    """

    K.clear_session()

    input_dim = len(wine_data.x_train[0])
    target_dim = len(wine_data.y_train[0])

    model = Sequential()
    model.add(Dense(input_dim, activation="relu"))
    model.add(Dense(input_dim * 3, activation="relu"))
    model.add(Dense(target_dim, activation="linear"))

    model.compile(optimizer="nadam", loss="mae")

    train_input = np.array(wine_data.x_train)
    train_target = np.array(wine_data.y_train)
    validation_input = np.array(wine_data.x_validate)
    validation_target = np.array(wine_data.y_validate)

    # Save model on improve of val_loss w/ Callback
    mc_c = ModelCheckpoint(filepath=f"models/{wine_data.wine_type}_ann_test.h5",
                           save_best_only=True,
                           save_weights_only=False,
                           verbose=1)

    # Early Stopping
    es_c = EarlyStopping(patience=25,
                         verbose=1)

    model.fit(x=train_input, y=train_target,
              validation_data=(validation_input, validation_target),
              batch_size=4,
              epochs=500,
              verbose=1,
              callbacks=[mc_c, es_c])


def main():

    wine_data = get_wine_data(wine_type="white", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
                              train_split_groups=[0, 1, 2], validation_split_groups=[3], test_split_groups=[4],
                              normalize=False)

    train_test_ann(wine_data=wine_data)

    wine_data = get_wine_data(wine_type="red", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
                              train_split_groups=[0, 1, 2], validation_split_groups=[3], test_split_groups=[4],
                              normalize=False)

    train_test_ann(wine_data=wine_data)


if __name__ == "__main__":

    main()


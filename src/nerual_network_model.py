from dataclasses import dataclass
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Nadam, SGD
import numpy as np
import csv
import keras.backend as K

from typing import List, Any, Dict
from sample_allocation import get_wine_data, WineData, QualityLabels, CHEM_ATTR_KEYS

"""
Cody Whitt
pkz325
CPSC 4240 Spring 2023
Final Project

ANN Model stuff. Currently very basic, but will be made dynamic in terms of pseudo-grid search / cross-validation
type stuff (similar to sample_allocation.py stuff).
"""


@dataclass
class ANNLayer:
    nodes: int
    activation: str


@dataclass
class ANNInfo:
    ann_id: int
    layers: List[ANNLayer]
    optimizer: str

    @staticmethod
    def from_dict(ann_info_dict: Dict):
        """
        For use w/ ANN GridSearch .csv
        """

        ann_id = ann_info_dict["ann_id"]
        optimizer = ann_info_dict["optimizer"]
        hidden_layer_1 = ANNLayer(nodes=ann_info_dict["hidden_layer_1_nodes"],
                                  activation=ann_info_dict["hidden_layer_1_activation"])

        hidden_layers = [hidden_layer_1]

        if ann_info_dict["hidden_layer_2_nodes"]:
            hidden_layer_2 = ANNLayer(nodes=ann_info_dict["hidden_layer_2_nodes"],
                                      activation=ann_info_dict["hidden_layer_2_activation"])
            hidden_layers.append(hidden_layer_2)

        return ANNInfo(ann_id=ann_id, layers=hidden_layers, optimizer=optimizer)


def train_test_ann(wine_data: WineData, ann_info: ANNInfo, for_grid_search: bool = False):
    """
    Doc Doc Doc
    """

    K.clear_session()
    np.random.seed(1337)

    input_dim = len(wine_data.x_train[0])
    target_dim = len(wine_data.y_train[0])

    model = Sequential()
    for i, ann_layer in enumerate(ann_info.layers):
        if i == 0:
            model.add(Dense(ann_layer.nodes, input_dim=input_dim, activation=ann_layer.activation))
        else:
            model.add(Dense(ann_layer.nodes, activation=ann_layer.activation))
    model.add(Dense(target_dim, activation="linear"))

    model.compile(optimizer=ann_info.optimizer, loss="mae")

    train_input = np.array(wine_data.x_train)
    train_target = np.array(wine_data.y_train)
    validation_input = np.array(wine_data.x_validate)
    validation_target = np.array(wine_data.y_validate)

    # Save model on improve of val_loss w/ Callback
    model_file_path = f"models/{wine_data.wine_type}_ann_test.h5"
    if for_grid_search:
        if wine_data.normalized:
            norm_str = "f_norm"
        else:
            norm_str = "f_raw"
        model_file_path = f"ann_grid_search/ags_models/{wine_data.wine_type}_{norm_str}_{ann_info.ann_id}.h5"

    mc_c = ModelCheckpoint(filepath=model_file_path,
                           save_best_only=True,
                           save_weights_only=False,
                           verbose=1,
                           )

    # Early Stopping
    es_c = EarlyStopping(patience=25,
                         verbose=1)

    model.fit(x=train_input, y=train_target,
              validation_data=(validation_input, validation_target),
              epochs=500,
              verbose=1,
              callbacks=[mc_c, es_c])

    if for_grid_search:

        train_pred_ann = model.predict(wine_data.x_train)
        validate_pred_ann = model.predict(wine_data.x_validate)
        test_pred_ann = model.predict(wine_data.x_test)
        ann_performance = wine_data.get_prediction_abs_error(train_pred=train_pred_ann, validate_pred=validate_pred_ann,
                                                             test_pred=test_pred_ann, tag=model_file_path)

        with open("ann_grid_search/ANNGridSearch_results.csv", "a") as a_file:
            writer = csv.writer(a_file, delimiter=",")
            writer.writerow([ann_info.ann_id, wine_data.wine_type, wine_data.normalized] + ann_performance.get_mae_pc())


def main():

    pass

    # wine_data = get_wine_data(wine_type="white", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
    #                           train_split_groups=[0, 1, 2], validation_split_groups=[3], test_split_groups=[4],
    #                           normalize=False)
    #
    # train_test_ann(wine_data=wine_data)
    #
    # wine_data = get_wine_data(wine_type="red", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
    #                           train_split_groups=[0, 1, 2], validation_split_groups=[3], test_split_groups=[4],
    #                           normalize=False)
    #
    # train_test_ann(wine_data=wine_data)


if __name__ == "__main__":

    main()


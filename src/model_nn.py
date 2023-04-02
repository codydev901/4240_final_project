from dataclasses import dataclass
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Nadam, SGD
import numpy as np
import csv
import keras.backend as K

from typing import List, Any, Dict
from load_allocate import get_wine_data, WineData, QualityLabels, CHEM_ATTR_KEYS

"""
Cody Whitt
pkz325
CPSC 4240 Spring 2023
Final Project

Doc Doc Doc
"""


@dataclass
class ANNLayer:  # Helper for dynamic NN structure
    nodes: int
    activation: str


@dataclass
class ANNInfo:  # Helper for dynamic NN structure
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

    def get_grid_info(self):

        if len(self.layers) == 2:
            layer_two = [self.layers[1].nodes, self.layers[1].activation]
        else:
            layer_two = [0, "NA"]

        return [self.optimizer, self.layers[0].nodes, self.layers[0].activation] + layer_two


# Final Configurations From Grid Search
SHALLOW_ANN_INFO = ANNInfo.from_dict({"ann_id": 4,
                                      "optimizer": "nadam",
                                      "hidden_layer_1_nodes": 96,
                                      "hidden_layer_1_activation": "relu",
                                      "hidden_layer_2_nodes": "",
                                      "hidden_layer_2_activation": ""})

DEEP_ANN_INFO = ANNInfo.from_dict({"ann_id": 64,
                                   "optimizer": "nadam",
                                   "hidden_layer_1_nodes": 96,
                                   "hidden_layer_1_activation": "relu",
                                   "hidden_layer_2_nodes": 48,
                                   "hidden_layer_2_activation": "relu"})


def train_test_ann(wine_data: WineData, ann_info: ANNInfo, for_grid_search: bool = False,
                   for_cv: bool = False, reduc_f: bool = False):
    """
    Doc Doc Doc
    """

    K.clear_session()
    np.random.seed(1337)

    input_dim = len(wine_data.x_train[0])
    target_dim = len(wine_data.y_train[0])

    model = Sequential()
    is_shallow = True
    for i, ann_layer in enumerate(ann_info.layers):
        if i == 0:
            model.add(Dense(ann_layer.nodes, input_dim=input_dim, activation=ann_layer.activation))
        else:
            model.add(Dense(ann_layer.nodes, activation=ann_layer.activation))
            is_shallow = False

    model.add(Dense(target_dim, activation="linear"))

    model.compile(optimizer=ann_info.optimizer, loss="mae")

    train_input = np.array(wine_data.x_train)
    train_target = np.array(wine_data.y_train)
    validation_input = np.array(wine_data.x_validate)
    validation_target = np.array(wine_data.y_validate)

    if wine_data.normalized:
        norm_str = "f_norm"
    else:
        norm_str = "f_raw"

    # Save model on improve of val_loss w/ Callback
    model_file_path = f"models/{wine_data.wine_type}_ann_test.h5"
    if for_grid_search:
        model_file_path = f"nn_grid_search/ags_models/{wine_data.wine_type}_{norm_str}_{ann_info.ann_id}.h5"
    if for_cv:
        if is_shallow:
            deep_shallow = "nn_shallow"
        else:
            deep_shallow = "nn_deep"

        if reduc_f:
            model_file_path = f"cross_validation/models_feature_reduc/{deep_shallow}/{wine_data.wine_type}_{norm_str}_reducf_val{wine_data.validate_groups[0]}_test{wine_data.test_groups[0]}.h5"
        else:
            model_file_path = f"cross_validation/models_full/{deep_shallow}/{wine_data.wine_type}_{norm_str}_allf_val{wine_data.validate_groups[0]}_test{wine_data.test_groups[0]}.h5"

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
              epochs=1000,
              verbose=1,
              callbacks=[mc_c, es_c])

    if for_grid_search:

        K.clear_session()
        model = load_model(model_file_path)

        train_pred_ann = model.predict(wine_data.x_train)
        validate_pred_ann = model.predict(wine_data.x_validate)
        test_pred_ann = model.predict(wine_data.x_test)
        ann_performance = wine_data.get_prediction_abs_error(train_pred=train_pred_ann, validate_pred=validate_pred_ann,
                                                             test_pred=test_pred_ann, tag=model_file_path)

        with open("nn_grid_search/ANNGridSearch_results.csv", "a") as a_file:
            writer = csv.writer(a_file, delimiter=",")
            writer.writerow([ann_info.ann_id, wine_data.wine_type, wine_data.normalized] + ann_info.get_grid_info() + ann_performance.get_mae_pc())

    if for_cv:

        K.clear_session()
        model = load_model(model_file_path)

        train_pred_ann = model.predict(wine_data.x_train)
        validate_pred_ann = model.predict(wine_data.x_validate)
        test_pred_ann = model.predict(wine_data.x_test)
        ann_performance = wine_data.get_prediction_abs_error(train_pred=train_pred_ann, validate_pred=validate_pred_ann,
                                                             test_pred=test_pred_ann, tag=model_file_path)

        if reduc_f:
            if is_shallow:
                with open("cross_validation/cv_ann_shallow_reduc.csv", "a") as a_file:
                    writer = csv.writer(a_file, delimiter=",")
                    writer.writerow(["nn_shallow", wine_data.wine_type, wine_data.normalized, "reduced",
                                     wine_data.validate_groups[0],
                                     wine_data.test_groups[0]] + ann_performance.get_mae_pc())
            else:
                with open("cross_validation/cv_ann_deep_reduc.csv", "a") as a_file:
                    writer = csv.writer(a_file, delimiter=",")
                    writer.writerow(["nn_deep", wine_data.wine_type, wine_data.normalized, "reduced",
                                     wine_data.validate_groups[0],
                                     wine_data.test_groups[0]] + ann_performance.get_mae_pc())
        else:
            if is_shallow:
                with open("cross_validation/cv_ann_shallow.csv", "a") as a_file:
                    writer = csv.writer(a_file, delimiter=",")
                    writer.writerow(["nn_shallow", wine_data.wine_type, wine_data.normalized, "all",
                                     wine_data.validate_groups[0], wine_data.test_groups[0]] + ann_performance.get_mae_pc())
            else:
                with open("cross_validation/cv_ann_deep.csv", "a") as a_file:
                    writer = csv.writer(a_file, delimiter=",")
                    writer.writerow(["nn_deep", wine_data.wine_type, wine_data.normalized, "all",
                                     wine_data.validate_groups[0], wine_data.test_groups[0]] + ann_performance.get_mae_pc())


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


from dataclasses import dataclass
from keras.models import load_model
from xgboost import XGBRegressor
from sample_allocation import get_wine_data, WineData, QualityLabels, CHEM_ATTR_KEYS
from typing import List

"""
Cody Whitt
pkz325
CPSC 4240 Spring 2023
Final Project

Doc Doc Doc
"""


def compare_model_performance(wine_data: WineData):

    # Load Models
    ann_model_fn = "white_ann_test.h5" if wine_data.wine_type == "white" else "red_ann_test.h5"
    xgboost_model_fn = "white_xgboost_test.json" if wine_data.wine_type == "white" else "red_xgboost_test.json"

    ann_model = load_model(f"models/{ann_model_fn}")

    xgboost_model = XGBRegressor()
    xgboost_model.load_model(fname=f"models/{xgboost_model_fn}")

    # Perform predictions on the three allocations
    train_pred_ann = ann_model.predict(wine_data.x_train)
    train_pred_xg = xgboost_model.predict(wine_data.x_train)
    train_pred_xg = [[v] for v in list(train_pred_xg)]

    validate_pred_ann = ann_model.predict(wine_data.x_validate)
    validate_pred_xg = xgboost_model.predict(wine_data.x_validate)
    validate_pred_xg = [[v] for v in list(validate_pred_xg)]

    test_pred_ann = ann_model.predict(wine_data.x_test)
    test_pred_xg = xgboost_model.predict(wine_data.x_test)
    test_pred_xg = [[v] for v in list(test_pred_xg)]

    ann_performance = wine_data.get_prediction_abs_error(train_pred=train_pred_ann, validate_pred=validate_pred_ann,
                                                         test_pred=test_pred_ann, tag=ann_model_fn)
    xgboost_performance = wine_data.get_prediction_abs_error(train_pred=train_pred_xg, validate_pred=validate_pred_xg,
                                                             test_pred=test_pred_xg, tag=xgboost_model_fn)

    ann_performance.show_info()
    xgboost_performance.show_info()


def main():

    white_wine_data = get_wine_data(wine_type="white", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
                                    train_split_groups=[0, 1, 2], validation_split_groups=[3], test_split_groups=[4],
                                    normalize=False)

    compare_model_performance(wine_data=white_wine_data)

    red_wine_data = get_wine_data(wine_type="red", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
                                  train_split_groups=[0, 1, 2], validation_split_groups=[3], test_split_groups=[4],
                                  normalize=False)

    compare_model_performance(wine_data=red_wine_data)


if __name__ == "__main__":

    main()

import numpy as np
import csv
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sample_allocation import get_wine_data, WineData, QualityLabels, CHEM_ATTR_KEYS

"""
Cody Whitt
pkz325
CPSC 4240 Spring 2023
Final Project

XGBoost Model stuff. Currently very basic, but will be made dynamic in terms of pseudo-grid search / cross-validation
type stuff (similar to sample_allocation.py stuff).
"""

LEARNING_RATE = [0.3, 0.03, 0.01]
MAX_DEPTH = [6, 12, 18]
N_ESTIMATORS = [100, 500, 1000]
GAMMA = [0, 0.1, 0.2]
SUBSAMPLE = [1.0, 0.75, 0.5]
MIN_CHILD_WEIGHT = [1.0, 5.0, 10.0]


def train_test_xgboost_grid_search(wine_data: WineData, learning_rate: float, max_depth: int, n_estimators: float,
                                   gamma: float, subsample: float, min_child_weight: float,
                                   grid_search_id: int):

    model = XGBRegressor(seed=1337, learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators,
                         gamma=gamma, subsample=subsample, min_child_weight=min_child_weight)

    model.fit(wine_data.x_train, wine_data.y_train, eval_metric='mae',
              eval_set=[(wine_data.x_validate, wine_data.y_validate)], verbose=False,
              early_stopping_rounds=25)

    if wine_data.normalized:
        norm_str = "f_norm"
    else:
        norm_str = "f_raw"

    save_path = f"xg_grid_search/xgg_models/{wine_data.wine_type}_{norm_str}_{grid_search_id}.json"

    train_pred_xg = model.predict(wine_data.x_train)
    train_pred_xg = [[v] for v in list(train_pred_xg)]

    validate_pred_xg = model.predict(wine_data.x_validate)
    validate_pred_xg = [[v] for v in list(validate_pred_xg)]

    test_pred_xg = model.predict(wine_data.x_test)
    test_pred_xg = [[v] for v in list(test_pred_xg)]

    xgboost_performance = wine_data.get_prediction_abs_error(train_pred=train_pred_xg, validate_pred=validate_pred_xg,
                                                             test_pred=test_pred_xg, tag=save_path)

    with open("xg_grid_search/XGGridSearch_results.csv", "a") as a_file:
        writer = csv.writer(a_file, delimiter=",")
        writer.writerow([grid_search_id, wine_data.wine_type,
                         wine_data.normalized] + [learning_rate, max_depth, n_estimators, gamma, subsample, min_child_weight] + xgboost_performance.get_mae_pc())

    model.save_model(fname=save_path)


def perform_xg_boost_grid_search_on_wine_data(wine_data: WineData):

    param_id = 0
    for lr in LEARNING_RATE:
        for md in MAX_DEPTH:
            for ne in N_ESTIMATORS:
                for ga in GAMMA:
                    for ss in SUBSAMPLE:
                        for mw in MIN_CHILD_WEIGHT:
                            train_test_xgboost_grid_search(wine_data=wine_data, learning_rate=lr, max_depth=md,
                                                           n_estimators=ne, gamma=ga, subsample=ss, min_child_weight=mw,
                                                           grid_search_id=param_id)
                            param_id += 1
                            print(param_id)


def main():

    white_wine_data = get_wine_data(wine_type="white", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
                                    train_split_groups=[0, 1, 2], validation_split_groups=[3], test_split_groups=[4],
                                    normalize=False)

    red_wine_data = get_wine_data(wine_type="red", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
                                  train_split_groups=[0, 1, 2], validation_split_groups=[3], test_split_groups=[4],
                                  normalize=False)

    white_wine_data_norm = get_wine_data(wine_type="white", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
                                         train_split_groups=[0, 1, 2], validation_split_groups=[3],
                                         test_split_groups=[4],
                                         normalize=True)

    red_wine_data_norm = get_wine_data(wine_type="red", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
                                       train_split_groups=[0, 1, 2], validation_split_groups=[3], test_split_groups=[4],
                                       normalize=True)

    perform_xg_boost_grid_search_on_wine_data(white_wine_data)
    perform_xg_boost_grid_search_on_wine_data(red_wine_data)
    perform_xg_boost_grid_search_on_wine_data(white_wine_data_norm)
    perform_xg_boost_grid_search_on_wine_data(red_wine_data_norm)


if __name__ == "__main__":

    main()

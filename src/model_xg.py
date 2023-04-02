import numpy as np
import csv
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from load_allocate import get_wine_data, WineData, QualityLabels, CHEM_ATTR_KEYS

"""
Cody Whitt
pkz325
CPSC 4240 Spring 2023
Final Project

XGBoost Model stuff. Configuration hard-coded to use the selected parameters chosen by grid search.

Models trained here used in the final comparison.
"""


def train_test_xgboost(wine_data: WineData, for_cv: bool = False):
    """
    Params set to be result from grid_search: xg_id == 381
    """

    model = XGBRegressor(seed=1337, learning_rate=0.03, max_depth=12, n_estimators=1000,
                         gamma=0.0, subsample=0.75, min_child_weight=1.0)

    model.fit(wine_data.x_train, wine_data.y_train, eval_metric='mae',
              eval_set=[(wine_data.x_validate, wine_data.y_validate)], verbose=False,
              early_stopping_rounds=25)

    if wine_data.normalized:
        norm_str = "f_norm"
    else:
        norm_str = "f_raw"

    if for_cv:

        save_path = f"cross_validation/models_full/xgboost/{wine_data.wine_type}_{norm_str}_allf_val{wine_data.validate_groups[0]}_test{wine_data.test_groups[0]}.json"

        train_pred_xg = model.predict(wine_data.x_train)
        train_pred_xg = [[v] for v in list(train_pred_xg)]

        validate_pred_xg = model.predict(wine_data.x_validate)
        validate_pred_xg = [[v] for v in list(validate_pred_xg)]

        test_pred_xg = model.predict(wine_data.x_test)
        test_pred_xg = [[v] for v in list(test_pred_xg)]

        xgboost_performance = wine_data.get_prediction_abs_error(train_pred=train_pred_xg,
                                                                 validate_pred=validate_pred_xg,
                                                                 test_pred=test_pred_xg, tag=save_path)

        with open("cross_validation/cv_xgboost.csv", "a") as a_file:
            writer = csv.writer(a_file, delimiter=",")
            writer.writerow(["xgboost", wine_data.wine_type,
                             wine_data.normalized, "all", wine_data.validate_groups[0], wine_data.test_groups[0]] + xgboost_performance.get_mae_pc())

        model.save_model(fname=save_path)


def main():

    wine_data = get_wine_data(wine_type="white", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
                              train_split_groups=[0, 1, 2], validation_split_groups=[3], test_split_groups=[4],
                              normalize=False)

    train_test_xgboost(wine_data=wine_data)

    # wine_data = get_wine_data(wine_type="red", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
    #                           train_split_groups=[0, 1, 2], validation_split_groups=[3], test_split_groups=[4],
    #                           normalize=False)
    #
    # train_test_xgboost(wine_data=wine_data)


if __name__ == "__main__":

    main()


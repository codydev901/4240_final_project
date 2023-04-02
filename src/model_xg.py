import numpy as np
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


def train_test_xgboost(wine_data: WineData):

    model = XGBRegressor(seed=1337, n_estimators=100, max_depth=6, learning_rate=0.3, subsample=1.0,
                         gamma=0.0, min_child_weight=10.0)

    model.fit(wine_data.x_train, wine_data.y_train, eval_metric='mae',
              eval_set=[(wine_data.x_validate, wine_data.y_validate)], verbose=True,
              early_stopping_rounds=25)

    validation_pred = list(model.predict(wine_data.x_validate))
    validation_error = [abs(v1[0] - v2) for v1, v2 in zip(wine_data.y_validate, validation_pred)]
    #
    print("xg boost, validation mean error test - for param tuning", np.mean(validation_error))
    #
    # model.save_model(fname=f"models/{wine_data.wine_type}_xgboost_test.json")


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


from xgboost import XGBRegressor

from sample_allocation import get_wine_data, WineData, QualityLabels, CHEM_ATTR_KEYS

"""
Cody Whitt
pkz325
CPSC 4240 Spring 2023
Final Project

XGBoost Model stuff. Currently very basic, but will be made dynamic in terms of pseudo-grid search / cross-validation
type stuff (similar to sample_allocation.py stuff).
"""


def train_test_xgboost(wine_data: WineData):

    model = XGBRegressor(seed=1337, objective='reg:absoluteerror')

    model.fit(wine_data.x_train, wine_data.y_train)

    model.save_model(fname=f"models/{wine_data.wine_type}_xgboost_test.json")


def main():

    wine_data = get_wine_data(wine_type="white", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
                              train_split_groups=[0, 1, 2], validation_split_groups=[3], test_split_groups=[4],
                              normalize=False)

    train_test_xgboost(wine_data=wine_data)

    wine_data = get_wine_data(wine_type="red", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
                              train_split_groups=[0, 1, 2], validation_split_groups=[3], test_split_groups=[4],
                              normalize=False)

    train_test_xgboost(wine_data=wine_data)


if __name__ == "__main__":

    main()


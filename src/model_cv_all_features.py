from copy import deepcopy
from typing import List
from load_allocate import get_wine_data, QualityLabels, CHEM_ATTR_KEYS

from model_xg import train_test_xgboost


def perform_cross_validation_training(train_groups: List[int], validation_groups: List[int],
                                      test_groups: List[int]):

    white_wine_data = get_wine_data(wine_type="white", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
                                    train_split_groups=train_groups, validation_split_groups=validation_groups,
                                    test_split_groups=test_groups, normalize=False)

    red_wine_data = get_wine_data(wine_type="red", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
                                  train_split_groups=train_groups, validation_split_groups=validation_groups,
                                  test_split_groups=test_groups, normalize=False)

    white_wine_data_norm = get_wine_data(wine_type="white", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
                                         train_split_groups=train_groups, validation_split_groups=validation_groups,
                                         test_split_groups=test_groups,
                                         normalize=True)

    red_wine_data_norm = get_wine_data(wine_type="red", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
                                       train_split_groups=train_groups, validation_split_groups=validation_groups,
                                       test_split_groups=test_groups, normalize=True)

    # XG Boost
    train_test_xgboost(wine_data=white_wine_data, for_cv=True)
    train_test_xgboost(wine_data=red_wine_data, for_cv=True)
    train_test_xgboost(wine_data=white_wine_data_norm, for_cv=True)
    train_test_xgboost(wine_data=red_wine_data_norm, for_cv=True)


def main():

    print("Running Cross-Validation on Full Feature")
    sample_groups = [0, 1, 2, 3, 4]
    for test_set in sample_groups:
        for validation_set in sample_groups:
            if test_set == validation_set:
                continue
            train_set = deepcopy(sample_groups)
            train_set.remove(test_set)
            train_set.remove(validation_set)
            print(train_set, [validation_set], [test_set])
            perform_cross_validation_training(train_groups=train_set, validation_groups=[validation_set],
                                              test_groups=[test_set])


if __name__ == "__main__":

    main()

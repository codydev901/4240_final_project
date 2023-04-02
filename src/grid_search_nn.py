import csv
from load_allocate import get_wine_data, QualityLabels, CHEM_ATTR_KEYS
from model_nn import ANNInfo, train_test_ann

"""
Cody Whitt
pkz325
CPSC 4240 Spring 2023
Final Project

Performs a grid search using the hyper-parameters defined in /nn_grid_search/ANNGridSearch_ann_info.csv

30 different shallow NN's will be trained, and 60 different deep NN's. With variation on optimizer, node number, and
activation functions.

The model configuration with the lowest average mean absolute error on the withheld test portion of the data across
the four (wine_type, normalized) inputs will be selected to represent NN in comparison with XGBoost.

TODO: Inline Comments
"""


def load_ann_grid_search_csv():
    """
    Loads the nn_grid_search/ANNGridSearch_ann_info.csv into memory for use in this experiment.
    """

    ann_info_grid = []

    with open("nn_grid_search/ANNGridSearch_ann_info.csv", "r") as r_file:
        reader = csv.DictReader(r_file, delimiter=",")
        for row in reader:
            ann_info_grid.append(ANNInfo.from_dict(row))

    print(f"Got {len(ann_info_grid)} ANNInfo For Grid Search")

    return ann_info_grid


def main():

    pass

    # Note: Commented out to avoid re-running (took around 3 hours locally)

    # ann_grid_info = load_ann_grid_search_csv()
    #
    # white_wine_data = get_wine_data(wine_type="white", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
    #                                 train_split_groups=[0, 1, 2], validation_split_groups=[3], test_split_groups=[4],
    #                                 normalize=False)
    #
    # red_wine_data = get_wine_data(wine_type="red", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
    #                               train_split_groups=[0, 1, 2], validation_split_groups=[3], test_split_groups=[4],
    #                               normalize=False)
    #
    # white_wine_data_norm = get_wine_data(wine_type="white", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
    #                                      train_split_groups=[0, 1, 2], validation_split_groups=[3], test_split_groups=[4],
    #                                      normalize=True)
    #
    # red_wine_data_norm = get_wine_data(wine_type="red", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
    #                                    train_split_groups=[0, 1, 2], validation_split_groups=[3], test_split_groups=[4],
    #                                    normalize=True)

    # for i, ann_info in enumerate(ann_grid_info):
    #
    #     train_test_ann(wine_data=white_wine_data, ann_info=ann_info, for_grid_search=True)
    #     train_test_ann(wine_data=red_wine_data, ann_info=ann_info, for_grid_search=True)
    #     train_test_ann(wine_data=white_wine_data_norm, ann_info=ann_info, for_grid_search=True)
    #     train_test_ann(wine_data=red_wine_data_norm, ann_info=ann_info, for_grid_search=True)


if __name__ == "__main__":

    main()

import csv
from load_allocate import get_wine_data, QualityLabels, CHEM_ATTR_KEYS
from model_nn import ANNInfo, train_test_ann

"""
Experiment attempting to find an ideal shallow and deep neural network configuration for use in this project in
comparison to XGBoost. 

30 different shallow and 60 deep networks will be trained/tested - with variation on optimizer, nodes, and activations.

These 90 variations are trained on both red/white, and normalized/raw. For 360 total models.

The best of the shallow and deep network configurations will be used in the the comparison, and this will also provide
insight into the effects of feature normalization.
"""


def load_ann_grid_search_csv():
    """
    Loads the ann_grid_search/ANNGridSearch_ann_info.csv into memory for use in this experiment.
    """

    ann_info_grid = []

    with open("ann_grid_search/ANNGridSearch_ann_info.csv", "r") as r_file:
        reader = csv.DictReader(r_file, delimiter=",")
        for row in reader:
            ann_info_grid.append(ANNInfo.from_dict(row))

    print(f"Got {len(ann_info_grid)} ANNInfo For Grid Search")

    return ann_info_grid


def main():

    ann_grid_info = load_ann_grid_search_csv()

    white_wine_data = get_wine_data(wine_type="white", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
                                    train_split_groups=[0, 1, 2], validation_split_groups=[3], test_split_groups=[4],
                                    normalize=False)

    red_wine_data = get_wine_data(wine_type="red", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
                                  train_split_groups=[0, 1, 2], validation_split_groups=[3], test_split_groups=[4],
                                  normalize=False)

    white_wine_data_norm = get_wine_data(wine_type="white", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
                                         train_split_groups=[0, 1, 2], validation_split_groups=[3], test_split_groups=[4],
                                         normalize=True)

    red_wine_data_norm = get_wine_data(wine_type="red", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
                                       train_split_groups=[0, 1, 2], validation_split_groups=[3], test_split_groups=[4],
                                       normalize=True)

    # for i, ann_info in enumerate(ann_grid_info):
    #
    #     train_test_ann(wine_data=white_wine_data, ann_info=ann_info, for_grid_search=True)
    #     train_test_ann(wine_data=red_wine_data, ann_info=ann_info, for_grid_search=True)
    #     train_test_ann(wine_data=white_wine_data_norm, ann_info=ann_info, for_grid_search=True)
    #     train_test_ann(wine_data=red_wine_data_norm, ann_info=ann_info, for_grid_search=True)


if __name__ == "__main__":

    main()

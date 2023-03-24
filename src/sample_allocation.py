import numpy as np
import pandas as pd
import itertools
from enum import Enum
from dataclasses import dataclass
from typing import List


"""
Cody Whitt
pkz325
CPSC 4240 Spring 2023
Final Project

Dynamic data wrapper/handler that returns wine quality features/labels in format ready to be used in the modeling 
steps. 

get_wine_data() function is meant to be used elsewhere, returns a WineData obj based on args.'

Randomization into train/validate/test groups performed. Randomization controlled by seed for overall 
reproducibility, but the train/validation/test groups may vary. All samples are shuffled and then dividing into 5
groups (each representing 20% of samples). Those 5 groups are then allocated as desired.
"""


class QualityLabels(Enum):
    RAW = "quality_raw"
    CLASS = "quality_class"
    CLASS_STR = "quality_class_str"


CHEM_ATTR_KEYS = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                  'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH',
                  'sulphates', 'alcohol']


@dataclass
class WineData:
    wine_type: str
    features: List[str]
    label: str
    x_train: List[List[float]]
    x_validate: List[List[float]]
    x_test: List[List[float]]
    y_train: List[List[float]]
    y_validate: List[List[float]]
    y_test: List[List[float]]
    train_groups: List[int]
    validate_groups: List[int]
    test_groups: List[int]
    train_sample_ids: List[int]
    validate_sample_ids: List[int]
    test_samples_ids: List[int]
    normalized: bool


def get_wine_data(wine_type: str,
                  features: List[str],
                  label: str,
                  train_split_groups: List[int],
                  validation_split_groups: List[int],
                  test_split_groups: List[int],
                  normalize: bool = False) -> WineData:
    """
    TODO: This is kinda slow, maybe rework or change initial parse format (add additional one)
    TODO: Potentially perform 60/20/20 split first grouped by quality
    TODO: inline comments
    TODO: Normalization
    """

    df = pd.read_csv("parsed_data/wine_combined_parsed.csv")
    df = df[df["wine_type"] == wine_type]

    all_sample_indices = list(df["sample_index"].unique())
    np.random.seed(1337)
    np.random.shuffle(all_sample_indices)

    allocation_groups = np.array_split(all_sample_indices, 5)

    train_indices = list(itertools.chain(*[list(v) for i, v in enumerate(allocation_groups)
                                           if i in train_split_groups]))
    validation_indices = list(itertools.chain(*[v for i, v in enumerate(allocation_groups)
                                                if i in validation_split_groups]))
    test_indices = list(itertools.chain(*[v for i, v in enumerate(allocation_groups) if i in test_split_groups]))

    x_train = []
    x_validate = []
    x_test = []
    y_train = []
    y_validate = []
    y_test = []

    for sample_index in all_sample_indices:
        sample_df = df[df["sample_index"] == sample_index]
        sample_features = []
        sample_labels = []
        for attr_key in features:
            sample_features.append(list(sample_df[sample_df["attr_key"] == attr_key]["attr_value"])[0])
        sample_labels.append(list(sample_df[label].unique())[0])
        if sample_index in validation_indices:
            x_validate.append(sample_features)
            y_validate.append(sample_labels)
        elif sample_index in test_indices:
            x_test.append(sample_features)
            y_test.append(sample_labels)
        else:
            x_train.append(sample_features)
            y_train.append(sample_labels)

    wine_data = WineData(wine_type=wine_type, features=features, label=label, x_train=x_train, x_validate=x_validate,
                         x_test=x_test, y_train=y_train, y_validate=y_validate, y_test=y_test,
                         train_groups=train_split_groups, validate_groups=validation_split_groups,
                         test_groups=test_split_groups, train_sample_ids=train_indices,
                         validate_sample_ids=validation_indices, test_samples_ids=test_indices,
                         normalized=normalize)

    return wine_data


def main():
    
    print("Wine Data Test")

    wine_data = get_wine_data(wine_type="red", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
                              train_split_groups=[0, 1, 2], validation_split_groups=[3], test_split_groups=[4],
                              normalize=False)

    print(wine_data)


if __name__ == "__main__":

    main()

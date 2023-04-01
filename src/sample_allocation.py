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
class WineModelPerformance:
    """
    TODO: Subclass this?
    """
    wine_type: str
    features: List[str]
    label: str
    tag: str
    train_abs_errors: List[float]
    validation_abs_errors: List[float]
    test_abs_errors: List[float]

    train_correct_quality_raw: List[bool]
    validation_correct_quality_raw: List[bool]
    test_correct_quality_raw: List[bool]

    def show_info(self):
        print("WineModelPerformance Info")
        print(self.wine_type)
        print(self.features)
        print(self.label)
        print(self.tag)
        print(f"Train      Error: {np.mean(self.train_abs_errors)}/{np.median(self.train_abs_errors)}/{np.max(self.train_abs_errors)}")
        print(f"Validation Error: {np.mean(self.validation_abs_errors)}/{np.median(self.validation_abs_errors)}/{np.max(self.validation_abs_errors)}")
        print(f"Test       Error: {np.mean(self.test_abs_errors)}/{np.median(self.test_abs_errors)}/{np.max(self.test_abs_errors)}")
        print(f"Train      Correct: {self.train_correct_quality_raw.count(True)}/{len(self.train_correct_quality_raw)}")
        print(f"Validation Correct: {self.validation_correct_quality_raw.count(True)}/{len(self.validation_correct_quality_raw)}")
        print(f"Test       Correct: {self.test_correct_quality_raw.count(True)}/{len(self.test_correct_quality_raw)}")

    def get_mae_pc(self):

        return [np.mean(self.train_abs_errors), np.mean(self.validation_abs_errors), np.mean(self.test_abs_errors),
                self.train_correct_quality_raw.count(True)/len(self.train_correct_quality_raw),
                self.validation_correct_quality_raw.count(True)/len(self.validation_correct_quality_raw),
                self.test_correct_quality_raw.count(True)/len(self.test_correct_quality_raw)]

    def get_friendly_name(self):
        return self.tag.split(".", 1)[0]


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

    def get_prediction_abs_error(self, train_pred: List[List[float]], validate_pred: List[List[float]],
                                 test_pred: List[List[float]], tag: str):

        train_diff = []
        train_correct = []
        for g_t, pred in zip(self.y_train, train_pred):
            abs_dff = abs(g_t[0] - pred[0])
            pred_round = round(pred[0])
            train_correct.append(int(g_t[0]) == pred_round)
            train_diff.append(abs_dff)

        validation_diff = []
        validation_correct = []
        for g_t, pred in zip(self.y_validate, validate_pred):
            abs_dff = abs(g_t[0] - pred[0])
            pred_round = round(pred[0])
            validation_correct.append(int(g_t[0]) == pred_round)
            validation_diff.append(abs_dff)

        test_diff = []
        test_correct = []
        for g_t, pred in zip(self.y_test, test_pred):
            abs_dff = abs(g_t[0] - pred[0])
            pred_round = round(pred[0])
            test_correct.append(int(g_t[0]) == pred_round)
            test_diff.append(abs_dff)

        return WineModelPerformance(wine_type=self.wine_type, features=self.features, label=self.label,
                                    tag=tag, train_abs_errors=train_diff, validation_abs_errors=validation_diff,
                                    test_abs_errors=test_diff, train_correct_quality_raw=train_correct,
                                    validation_correct_quality_raw=validation_correct,
                                    test_correct_quality_raw=test_correct)


def get_wine_data(wine_type: str,
                  features: List[str],
                  label: str,
                  train_split_groups: List[int],
                  validation_split_groups: List[int],
                  test_split_groups: List[int],
                  normalize: bool = False) -> WineData:
    """
    Dynamic X/Y loading for use in the modeling steps. Performs 60/20/20 split within each quality class to account
    for class imbalance. Group assignment controlled by args for cross-validation. Min/max normalization if desired.

    Works off of parsed_data/wine_combined_parsed_flat.csv
    """

    # Load and Filter
    df = pd.read_csv("parsed_data/wine_combined_parsed_flat.csv")
    df = df[df["wine_type"] == wine_type]

    # For feature/label data after assignment below
    x_train = []
    x_validate = []
    x_test = []
    y_train = []
    y_validate = []
    y_test = []

    # For overall sample index allocation
    overall_train_indices = []
    overall_validation_indices = []
    overall_test_indices = []

    # To apply a 60/20/20 split within quality classes
    raw_qualities = list(df["quality_raw"].unique())
    raw_qualities.sort()

    # For reproducibility
    np.random.seed(1337)

    # For use in Min/Max Normalization if Applicable
    feature_min = [df[f].min() for f in features]
    feature_max = [df[f].max() for f in features]

    for raw_quality in raw_qualities:

        # Subset DF on Quality
        sub_df = df[df["quality_raw"] == raw_quality]
        # Get list of sample_indices & Shuffle
        sub_sample_indices = list(sub_df["sample_index"].unique())
        np.random.shuffle(sub_sample_indices)

        # Split into 5 groups, assign to train/validation/test per function arguments
        allocation_groups = np.array_split(sub_sample_indices, 5)

        train_indices = list(itertools.chain(*[list(v) for i, v in enumerate(allocation_groups)
                                               if i in train_split_groups]))
        validation_indices = list(itertools.chain(*[v for i, v in enumerate(allocation_groups)
                                                    if i in validation_split_groups]))
        test_indices = list(itertools.chain(*[v for i, v in enumerate(allocation_groups) if i in test_split_groups]))

        # Iterate and Populate X/Y in context of above allocation.
        for i, row in sub_df.iterrows():

            sample_index = row["sample_index"]
            sample_features = [row[f] for f in features]  # Option to sub-select features from arg.

            # Normalize by min/max per arg
            if normalize:
                norm_sample_features = []
                for n_i, s_f in enumerate(sample_features):
                    f_max = feature_max[n_i]
                    f_min = feature_min[n_i]
                    f_norm = (s_f - f_min) / (f_max - f_min)
                    norm_sample_features.append(f_norm)
                sample_features = norm_sample_features

            sample_labels = [row[label]]

            # Add to X/Y
            if sample_index in validation_indices:
                x_validate.append(sample_features)
                y_validate.append(sample_labels)
            elif sample_index in test_indices:
                x_test.append(sample_features)
                y_test.append(sample_labels)
            else:
                x_train.append(sample_features)
                y_train.append(sample_labels)

        # Keep overall track of which samples went where
        overall_train_indices += train_indices
        overall_validation_indices += validation_indices
        overall_test_indices += test_indices

    wine_data = WineData(wine_type=wine_type, features=features, label=label, x_train=x_train, x_validate=x_validate,
                         x_test=x_test, y_train=y_train, y_validate=y_validate, y_test=y_test,
                         train_groups=train_split_groups, validate_groups=validation_split_groups,
                         test_groups=test_split_groups, train_sample_ids=overall_train_indices,
                         validate_sample_ids=overall_validation_indices, test_samples_ids=overall_test_indices,
                         normalized=normalize)

    return wine_data


def check_sample_class_balance(wine_type: str):
    """
    Quick look at sample/quality representation in the data.

    Most samples below to the "medium" group. When doing the train/validation/test split
    above, want to ensure that the split is performed within the quality group to prevent the less represented
    quality groups from potentially being over or under represented.
    """

    df = pd.read_csv("parsed_data/wine_combined_parsed_flat.csv")
    df = df[df["wine_type"] == wine_type]

    print(df.head())
    print(df.groupby(["quality_raw"]).count())


def main():

    print("Wine Data Test")

    wine_data = get_wine_data(wine_type="red", features=CHEM_ATTR_KEYS, label=QualityLabels.RAW.value,
                              train_split_groups=[0, 1, 2], validation_split_groups=[3], test_split_groups=[4],
                              normalize=False)

    print(wine_data)

    # check_sample_class_balance(wine_type="white")
    # check_sample_class_balance(wine_type="red")


if __name__ == "__main__":

    main()

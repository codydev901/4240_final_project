import pandas as pd
from dataclasses import dataclass
from keras.models import load_model
from xgboost import XGBRegressor
from sample_allocation import get_wine_data, WineData, QualityLabels, CHEM_ATTR_KEYS, WineModelPerformance
from typing import List
from load_and_parse import quality_class_from_quality

"""
Cody Whitt
pkz325
CPSC 4240 Spring 2023
Final Project

Doc Doc Doc
"""


def write_model_performance(wine_data: WineData, ann_performance: WineModelPerformance,
                            xg_performance: WineModelPerformance):

    print(f"Write Performance:{wine_data.wine_type}/{ann_performance.tag}/{xg_performance.tag}")

    res_df = [["sample_index", "sample_allocation", "wine_type", "quality_ground_truth", "quality_str",
               "ann_abs_error", "ann_is_correct", "xg_abs_error", "xg_is_correct"]]

    for i, sample_index in enumerate(wine_data.train_sample_ids):
        quality_g_t = wine_data.y_train[i][0]
        quality_str = quality_class_from_quality(int(quality_g_t), as_str=True)
        ann_abs_error = ann_performance.train_abs_errors[i]
        ann_is_correct = ann_performance.train_correct_quality_raw[i]
        xg_abs_error = xg_performance.train_abs_errors[i]
        xg_is_correct = xg_performance.train_correct_quality_raw[i]
        res_df.append([sample_index, "train", wine_data.wine_type, quality_g_t, quality_str,
                       ann_abs_error, ann_is_correct, xg_abs_error, xg_is_correct])

    for i, sample_index in enumerate(wine_data.validate_sample_ids):
        quality_g_t = wine_data.y_validate[i][0]
        quality_str = quality_class_from_quality(int(quality_g_t), as_str=True)
        ann_abs_error = ann_performance.validation_abs_errors[i]
        ann_is_correct = ann_performance.validation_correct_quality_raw[i]
        xg_abs_error = xg_performance.validation_abs_errors[i]
        xg_is_correct = xg_performance.validation_correct_quality_raw[i]
        res_df.append([sample_index, "validation", wine_data.wine_type, quality_g_t, quality_str,
                       ann_abs_error, ann_is_correct, xg_abs_error, xg_is_correct])

    for i, sample_index in enumerate(wine_data.test_samples_ids):
        quality_g_t = wine_data.y_test[i][0]
        quality_str = quality_class_from_quality(int(quality_g_t), as_str=True)
        ann_abs_error = ann_performance.test_abs_errors[i]
        ann_is_correct = ann_performance.test_correct_quality_raw[i]
        xg_abs_error = xg_performance.test_abs_errors[i]
        xg_is_correct = xg_performance.test_correct_quality_raw[i]
        res_df.append([sample_index, "test", wine_data.wine_type, quality_g_t, quality_str,
                       ann_abs_error, ann_is_correct, xg_abs_error, xg_is_correct])

    res_df = pd.DataFrame(data=res_df[1:], columns=res_df[0])
    res_df.to_csv(f"prediction_results/{wine_data.wine_type}_{ann_performance.get_friendly_name()}_{xg_performance.get_friendly_name()}_prediction_results.csv", index=False)


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

    write_model_performance(wine_data=wine_data, ann_performance=ann_performance, xg_performance=xgboost_performance)


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

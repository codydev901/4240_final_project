import pandas as pd
import numpy as np

"""
Doc Doc Doc
"""

WINE_TYPES = ["red", "white"]
NORMALIZED = [True, False]


def get_best_xg_model_params():

    xg_grid_df = pd.read_csv("xg_grid_search/XGGridSearch_results.csv")

    overall_test_mae_score = {}

    for wine_type in WINE_TYPES:
        for normalized in NORMALIZED:
            sub_df = xg_grid_df[(xg_grid_df["wine_type"] == wine_type) &
                                (xg_grid_df["normalized"] == normalized)]
            sub_df.sort_values(by=["test_mae"], inplace=True)
            print(sub_df.head(n=10))
            for i, row in sub_df.iterrows():
                xg_id = row["xg_id"]
                test_mae = row["test_mae"]
                try:
                    overall_test_mae_score[xg_id].append(test_mae)
                except KeyError:
                    overall_test_mae_score[xg_id] = [test_mae]

    overall_test_mae_score = [[k, np.mean(overall_test_mae_score[k])] for k in overall_test_mae_score]
    overall_test_mae_score.sort(key=lambda x: x[-1])
    print(overall_test_mae_score[:10])

    print(f"Best Overall XGBoost Model Param:{overall_test_mae_score[0]}")
    print(xg_grid_df.loc[xg_grid_df["xg_id"] == overall_test_mae_score[0][0]])


def get_best_ann_model_params():

    nn_grid_df = pd.read_csv("nn_grid_search/ANNGridSearch_results.csv")

    overall_test_mae_score_shallow = {}
    overall_test_mae_score_deep = {}

    for wine_type in WINE_TYPES:
        for normalized in NORMALIZED:
            sub_df = nn_grid_df[(nn_grid_df["wine_type"] == wine_type) &
                                (nn_grid_df["normalized"] == normalized)]
            sub_df.sort_values(by=["test_mae"], inplace=True)
            print(sub_df.head(n=10))
            for i, row in sub_df.iterrows():
                ann_id = row["ann_id"]
                test_mae = row["test_mae"]

                res_dict = overall_test_mae_score_shallow
                if row["hidden_layer_2_nodes"] != 0:
                    res_dict = overall_test_mae_score_deep

                try:
                    res_dict[ann_id].append(test_mae)
                except KeyError:
                    res_dict[ann_id] = [test_mae]

    # Shallow NN Result
    overall_test_mae_score_shallow = [[k, np.mean(overall_test_mae_score_shallow[k])] for k in overall_test_mae_score_shallow]
    overall_test_mae_score_shallow.sort(key=lambda x: x[-1])
    print(overall_test_mae_score_shallow[:10])
    print(f"Best Overall Shallow NN Model Param:{overall_test_mae_score_shallow[0]}")
    print(nn_grid_df.loc[nn_grid_df["ann_id"] == overall_test_mae_score_shallow[0][0]])

    # Deep NN Result
    overall_test_mae_score_deep = [[k, np.mean(overall_test_mae_score_deep[k])] for k in overall_test_mae_score_deep]
    overall_test_mae_score_deep.sort(key=lambda x: x[-1])
    print(overall_test_mae_score_deep[:10])
    print(f"Best Overall Deep NN Model Param:{overall_test_mae_score_deep[0]}")
    print(nn_grid_df.loc[nn_grid_df["ann_id"] == overall_test_mae_score_deep[0][0]])


def main():

    get_best_xg_model_params()

    get_best_ann_model_params()


if __name__ == "__main__":

    main()

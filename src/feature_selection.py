import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

from load_allocate import CHEM_ATTR_KEYS

"""
Doc Doc Doc
"""

REDUCED_FEATURES = ["alcohol", "volatile acidity", "density", "chlorides", "total sulfur dioxide", "sulphates"]


def perform_feature_correlation():

    df = pd.read_csv("parsed_data/wine_combined_parsed_flat.csv")

    res_df = [["wine_type", "chem_attr", "pearson", "spearman", "pearson_abs", "spearman_abs", "avg_abs"]]

    for wine_type in ["red", "white"]:
        sub_df = df[df["wine_type"] == wine_type]
        sub_df = sub_df[CHEM_ATTR_KEYS + ["quality_raw"]]

        y = list(sub_df["quality_raw"])
        for chem_attr in CHEM_ATTR_KEYS:
            x = list(sub_df[chem_attr])
            s_c = spearmanr(x, y)
            p_c = pearsonr(x, y)
            res_df.append([wine_type, chem_attr, p_c.statistic, s_c.statistic, abs(p_c.statistic), abs(s_c.statistic),
                           np.mean([abs(p_c.statistic), abs(s_c.statistic)])])

    res_df = pd.DataFrame(data=res_df[1:], columns=res_df[0])
    res_df.sort_values(by=["avg_abs"], inplace=True, ascending=False)
    print(res_df.head(n=22))

    avg_abs_correlation = res_df.groupby(["chem_attr"])["avg_abs"].mean().to_frame(name="avg_corr_abs").reset_index()
    avg_abs_correlation.sort_values(by=["avg_corr_abs"], inplace=True, ascending=False)
    print("Average Correlation For Chem Attributes")
    print(avg_abs_correlation)


def main():

    perform_feature_correlation()


if __name__ == "__main__":

    main()

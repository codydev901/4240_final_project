import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

from load_allocate import CHEM_ATTR_KEYS
import plotly.express as px

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


def perform_feature_correlation_two():

    df = pd.read_csv("parsed_data/wine_combined_parsed_flat.csv")

    res_df = [["Wine Type", "Feature", "Spearman Correlation To Quality"]]

    for wine_type in ["red", "white"]:
        sub_df = df[df["wine_type"] == wine_type]
        sub_df = sub_df[CHEM_ATTR_KEYS + ["quality_raw"]]

        y = list(sub_df["quality_raw"])
        for chem_attr in CHEM_ATTR_KEYS:
            x = list(sub_df[chem_attr])
            s_c = spearmanr(x, y)
            res_df.append([wine_type, chem_attr, s_c.statistic])

    res_df = pd.DataFrame(data=res_df[1:], columns=res_df[0])
    res_df.sort_values(by=["Spearman Correlation To Quality"], inplace=True, ascending=False)
    print(res_df.head(n=22))

    # avg_abs_correlation = res_df.groupby(["Feature"])["spearman"].mean().to_frame(name="Mean Spearman Correlation to Quality").reset_index()
    # avg_abs_correlation.sort_values(by=["Mean Spearman Correlation to Quality"], inplace=True, ascending=False)
    # print("Average Correlation For Chem Attributes")
    # print(avg_abs_correlation)

    fig = px.scatter(data_frame=res_df, x="Feature", y="Spearman Correlation To Quality", color="Wine Type",
                     title="Feature vs. Spearman Correlation To Quality")

    # fig.update_layout({
    #     'paper_bgcolor': 'rgba(235, 235, 235, 0)',
    # })

    fig.update_yaxes(title_font=dict(size=14), tickfont=dict(size=20))
    fig.update_xaxes(title_font=dict(size=14), tickfont=dict(size=20))
    fig.update_traces(marker={'size': 12})

    fig.show()


def main():

    # perform_feature_correlation()

    perform_feature_correlation_two()


if __name__ == "__main__":

    main()

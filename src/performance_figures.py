import plotly.express as px
import pandas as pd

"""
Doc Doc Doc
"""


def generate_performance_figures(prediction_result_file: str):

    df = pd.read_csv(f"prediction_results/{prediction_result_file}")

    wine_type = list(df["wine_type"].unique())[0].title()

    plot_df = [["wine_type", "allocation", "model_type", "metric_key", "metric_value"]]

    for allocation in list(df["sample_allocation"].unique()):
        a_df = df[df["sample_allocation"] == allocation]
        plot_df.append([wine_type, allocation, "shallow_mlp", "mean_abs_error", a_df["ann_abs_error"].mean()])
        plot_df.append([wine_type, allocation, "shallow_mlp", "percent_correct", a_df["ann_is_correct"].values.sum() / len(a_df)])
        plot_df.append([wine_type, allocation, "xgboost", "mean_abs_error", a_df["xg_abs_error"].mean()])
        plot_df.append([wine_type, allocation, "xgboost", "percent_correct", a_df["xg_is_correct"].values.sum() / len(a_df)])

    plot_df = pd.DataFrame(data=plot_df[1:], columns=plot_df[0])
    print(plot_df.head())

    mean_error_df = plot_df[plot_df["metric_key"] == "mean_abs_error"]
    fig = px.bar(data_frame=mean_error_df, x="allocation", y="metric_value", color="model_type",
                 title=f"{wine_type} Wine Prediction Mean Error", barmode="group", text_auto=".2f")
    fig.show()

    percent_correct_df = plot_df[plot_df["metric_key"] == "percent_correct"]
    fig = px.bar(data_frame=percent_correct_df, x="allocation", y="metric_value", color="model_type",
                 title=f"{wine_type} Prediction Percent Correct", barmode="group", text_auto=".2f")
    fig.show()

    # Make more dynamic/safe in future...
    plot_df.to_csv(f"figures/{wine_type}_performance_info.csv", index=False)


def main():

    generate_performance_figures(prediction_result_file="white_white_ann_test_white_xgboost_test_prediction_results.csv")
    generate_performance_figures(prediction_result_file="red_red_ann_test_red_xgboost_test_prediction_results.csv")


if __name__ == "__main__":

    main()

import pandas as pd
import plotly.express as px

"""
Doc Doc Doc
"""


def main():

    res_df = [["Model Type", "Test MAE", "Test Percent Correct", "Wine Type", "Feature Type"]]

    xg_df = pd.read_csv("xg_grid_search/XGGridSearch_results.csv")

    for i, row in xg_df.iterrows():

        feature_type = "raw" if not row["normalized"] else "normalized"

        res_df.append(["XGBoost", row["test_mae"], row["test_pc"], row["wine_type"], feature_type])

    nn_df = pd.read_csv("nn_grid_search/ANNGridSearch_results.csv")

    for i, row in nn_df.iterrows():

        feature_type = "raw" if not row["normalized"] else "normalized"
        model_type = "Shallow MLP" if row["hidden_layer_2_nodes"] == 0 else "Deep MLP"

        res_df.append([model_type, row["test_mae"], row["test_pc"], row["wine_type"], feature_type])

    res_df = pd.DataFrame(data=res_df[1:], columns=res_df[0])

    res_df = res_df[res_df["Test Percent Correct"] > .4]

    fig = px.scatter(data_frame=res_df, x="Test MAE", y="Test Percent Correct", color="Model Type",
                     facet_col="Wine Type", facet_col_wrap=2, facet_row="Feature Type",
                     title="Model HyperParameter GridSearch")

    # fig.update_layout({
    #     'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    #     'plot_bgcolor': 'rgba(255,255,255,0)',
    # })

    fig.update_yaxes(title_font=dict(size=14), tickfont=dict(size=20))
    fig.update_xaxes(title_font=dict(size=14), tickfont=dict(size=20))
    fig.update_annotations(font_size=20)

    fig.show()


if __name__ == "__main__":

    main()

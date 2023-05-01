import pandas as pd
import plotly.express as px


def main():

    res = [["Wine Type", "Model Type", "Feature State", "Test Percent Correct CV"]]

    all_df = []

    for cv_file in ["cv_ann_deep.csv", "cv_ann_deep_reduc.csv", "cv_ann_shallow.csv", "cv_ann_shallow_reduc.csv",
                    "cv_xgboost.csv", "cv_xgboost_reduc.csv"]:

        df = pd.read_csv(f"cross_validation/{cv_file}")
        all_df.append(df)

    all_df = pd.concat(all_df)
    df = all_df

    for model_type in list(df["model_type"].unique()):
        for wine_type in list(df["wine_type"].unique()):
            for reduc in ["all", "reduced"]:
                for norm in [False, True]:

                    sub_df = df[(df["model_type"] == model_type) &
                                (df["wine_type"] == wine_type) &
                                 (df["features"] == reduc) &
                                 (df["normalized"] == norm)]

                    reduc_str = reduc.title()
                    norm_str = "Normalized" if norm else "Raw"

                    feature_state = f"{reduc_str} {norm_str}"

                    res.append([wine_type,
                                model_type,
                                feature_state,
                                round(sub_df["test_pc"].mean()*100.0, 2)])

    res_df = pd.DataFrame(data=res[1:], columns=res[0])

    print(res_df.head(n=30))

    fig = px.bar(data_frame=res_df, x="Feature State", y="Test Percent Correct CV", color="Model Type",
                 facet_col="Wine Type", facet_col_wrap=2, title="Final Model Comparison - Cross Validation", barmode="group",
                 text_auto='.2f')

    # fig.update_layout({
    #     'paper_bgcolor': 'rgba(235, 235, 235, 0)',
    # })

    fig.update_yaxes(title_font=dict(size=14), tickfont=dict(size=24))
    fig.update_xaxes(title_font=dict(size=14), tickfont=dict(size=24))
    fig.update_annotations(font_size=24)

    for model_type in ["xgboost", "nn_deep", "nn_shallow"]:
        sub_df = res_df[res_df["Model Type"] == model_type]
        print(model_type, sub_df["Test Percent Correct CV"].mean())

    fig.show()


if __name__ == "__main__":

    main()

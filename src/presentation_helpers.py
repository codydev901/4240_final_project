import pandas as pd


def main():

    df = pd.read_csv("parsed_data/wine_combined_parsed_flat.csv")
    print(df.groupby(["wine_type", "quality_raw"]).count())


if __name__ == "__main__":

    main()

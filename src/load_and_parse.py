import pandas as pd


"""
Cody Whitt
pkz325
CPSC 4240 Spring 2023
Final Project

Load/Parse code for raw wine quality data sets. Gets them in a more accessible format, determines an additional quality 
classification metric, and ends up combining them together (separated back out later on).

Writes a .csv to /parsed_data
"""


def quality_class_from_quality(quality: int, as_str: bool):
    """
    Helper function.

    Assigns a classification based on the underlying quality score. 3 Groups.
    """

    if quality <= 4:
        return "low" if as_str else 0
    if quality <= 6:
        return "medium" if as_str else 1

    return "high" if as_str else 2


def load_parse_wine_csv(wine_csv_file_path: str):
    """
    Overall load/parse function w/ intermediate steps printed to console.

    TODO: Inline comments
    """

    wine_type = "red" if "red" in wine_csv_file_path else "white"

    raw_df = pd.read_csv(wine_csv_file_path, delimiter=";")
    print("Initial Load")
    print(raw_df.head())
    print(raw_df.info())

    print("Chem Attributes")
    chem_attrs = list(raw_df.columns[:-1])
    print(chem_attrs)

    res_df = [["sample_index", "wine_type", "attr_key", "attr_value", "quality_raw"]]
    for i, row in raw_df.iterrows():
        for chem_k in chem_attrs:
            res_df.append([i, wine_type, chem_k, row[chem_k], row["quality"]])

    res_df = pd.DataFrame(data=res_df[1:], columns=res_df[0])

    print("Unique Quality Raw")
    print(res_df.groupby(["quality_raw"]).count())

    res_df["quality_class_str"] = res_df["quality_raw"].apply(lambda x: quality_class_from_quality(x, as_str=True))
    res_df["quality_class"] = res_df["quality_raw"].apply(lambda x: quality_class_from_quality(x, as_str=False))

    print("With Quality Sub-Classes")
    print(res_df.head())
    print(res_df.info())

    return res_df


def main():

    red_file_path = "raw_data/winequality-red.csv"
    white_file_path = "raw_data/winequality-white.csv"

    red_df = load_parse_wine_csv(red_file_path)
    white_df = load_parse_wine_csv(white_file_path)

    df = pd.concat([red_df, white_df])
    df.to_csv("parsed_data/wine_combined_parsed.csv", index=False)


if __name__ == "__main__":

    main()

#%%
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import os
import re
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


today = pd.Timestamp.today().normalize()

ten_years_ago = today - pd.DateOffset(years=10)
five_years_ago = today - pd.DateOffset(years=5)
three_years_ago = today - pd.DateOffset(years=3)
two_years_ago = today - pd.DateOffset(years=2)
one_year_ago = today - pd.DateOffset(years=1)
beginning_of_cal_year = pd.Timestamp(today.year, 1, 1)
beginning_of_last_cal_year = pd.Timestamp(today.year - 1, 1, 1)


def get_latest_data_file(file_string: str, data_dir: str | Path = "data") -> str:
    """Gets requested data file from "data" directory

    file_string takes string of requested data file

    Returns directory last alphabetical file name containing file_string"""

    base = Path(data_dir)

    if not base.is_absolute():
        base = Path.cwd() / base
    if not base.exists():
        raise FileNotFoundError(f"Data Directory not found: {base}")

    dir_path = os.path.join(os.getcwd(), "data")
    file_list = [
        p
        for p in base.iterdir()
        if p.is_file() and file_string.lower() in p.name.lower()
    ]

    if not file_list:
        raise FileNotFoundError(f'No files in "{base}" containing "{file_string}"')

    file_list.sort(key=lambda p: p.name.lower(), reverse=True)
    return os.path.join(dir_path, file_list[0])


# Open patient file as dataframe
demographics_df = pd.read_csv(
    get_latest_data_file("demographics"), index_col="enterpriseid"
)

# Define dependent variables for sklearn
flag_cols = pd.DataFrame(columns=["No AWV",
               "No CRC", 
               "No BrC",
               "A1C >9",
               "BP >140/90"],
               index=demographics_df.index)

#%% ------------------- WITHOUT AWV -------------------

enc_base_df = pd.read_csv(
    get_latest_data_file("encounter base"),
    index_col="cln enc id",
    parse_dates=["cln enc date"],
)

# Filter appointments by wellness visits occuring on or after 1/1/2025
pattern = "|".join(["well", "awv"])

# Filter for encounters with appointment types containing "well" or "awv" (case insensitive) and date on or after 1/1/2025
enc_w_awv_df = enc_base_df.loc[
    (enc_base_df["appttype"].str.contains(pattern, case=False, na=False))
    & (enc_base_df["cln enc date"] >= beginning_of_cal_year)
]


encid_w_awv = enc_w_awv_df["enterpriseid"].dropna().unique()


# Get patients without wellness visits since 1/1/2025 and over the age of 40
flag_cols["No AWV"] = (~demographics_df.index.isin(encid_w_awv) & (demographics_df["age"] >= 40)).astype("int8")

display(flag_cols["No AWV"])


# %% ------------------- WITHOUT CRC -------------------

# Open surgical history file as dataframe
surghx_df = pd.read_csv(
    get_latest_data_file("surgical history"), parse_dates=["surg hist date"]
)

# Open cologuard file as dataframe
cologuard_df = pd.read_csv(get_latest_data_file("cologuard"), parse_dates=["labdate"])

# Filter all colonoscopies occurring in the past 10 years
colonos_df = surghx_df.loc[
    surghx_df["surg hist proc"].str.contains("colonosc", case=False, na=False)
    & (surghx_df["surg hist date"] >= ten_years_ago)
]

# Filter all flex sigmoidoscopies occurring in the past 5 years
flexsig_df = surghx_df.loc[
    surghx_df["surg hist proc"].str.contains("sigmoidosc", case=False, na=False)
    & (surghx_df["surg hist date"] >= five_years_ago)
]

# Filter all cologuards occurring in the past 3 years
cologua_df = cologuard_df.loc[(cologuard_df["labdate"] >= three_years_ago)]

ids_colonos = pd.Index(colonos_df["enterpriseid"].dropna().unique())
ids_flexsig = pd.Index(flexsig_df["enterpriseid"].dropna().unique())
ids_cologua = pd.Index(cologua_df["enterpriseid"].dropna().unique())

colo_complete = ids_colonos.union(ids_flexsig).union(ids_cologua)

flag_cols["No CRC"] = (~demographics_df.index.isin(colo_complete) & (demographics_df["age"] >= 45)).astype("int8")

display(flag_cols)

# %% ------------------- WITHOUT BrC -------------------

mammogram_df = pd.read_csv(
    get_latest_data_file("mammogram"), parse_dates=["dt f lst mmmgrm"]
)
mammogram_df["dt f lst mmmgrm"] = pd.to_datetime(
    mammogram_df["dt f lst mmmgrm"], errors="coerce"
)

recent_mammo_df = mammogram_df[mammogram_df["dt f lst mmmgrm"] >= two_years_ago]

flag_cols["No BrC"] = (~demographics_df.index.isin(recent_mammo_df["enterpriseid"])
    & (demographics_df["age"] >= 40)
    & (demographics_df["patientsex"] == "F")).astype("int8")

display(flag_cols)

# %% ------------------- A1C >9 -------------------
a1c_df = pd.read_csv(
    get_latest_data_file("a1c.csv"), parse_dates=["labdate"]
)

recent_a1c_df = a1c_df[a1c_df["labdate"] >= two_years_ago].copy()

# Filter lab results for A1C tests in the past 2 years
max_a1c_df = recent_a1c_df.groupby("enterpriseid")["labvalue"].max().reset_index()


# Get IDs of patients with max A1C >9
a1c_ov_9_ids = pd.Index(max_a1c_df.loc[max_a1c_df["labvalue"] > 9, "enterpriseid"])


flag_cols["A1C >9"] = (demographics_df.index.isin(a1c_ov_9_ids)).astype("int8")

display(flag_cols["A1C >9"].value_counts())


#%% ------------------- BP >140/90 -------------------
vitals_df = pd.read_csv(get_latest_data_file("vitals"))
enc_base_df = pd.read_csv(
    get_latest_data_file("encounter base"),
    parse_dates=["cln enc date"],
    index_col=["cln enc id"],
)

latest_enc_df = enc_base_df.sort_values(
    ["enterpriseid", "cln enc date"], ascending=[True, False]
).drop_duplicates(subset="enterpriseid", keep="first")

latest_vitals_df = pd.merge(vitals_df, latest_enc_df, on="cln enc id").sort_values(["cln enc date"])

bp_out_of_range_index = pd.Index(
    latest_vitals_df[
        ((latest_vitals_df["sys BP"] >= 140)|(latest_vitals_df["dia BP"] >= 90))
        & (latest_vitals_df["cln enc date"] >= one_year_ago)
    ]["enterpriseid"])

flag_cols["BP >140/90"] = (
    demographics_df.index.isin(bp_out_of_range_index)
).astype("int8")

display(flag_cols)

# %%

bmi_df = pd.read_csv(get_latest_data_file("vitals"))[["cln enc id", "enc BMI"]]
enc_base_df = pd.read_csv(
    get_latest_data_file("encounter base"),
    parse_dates=["cln enc date"],
    index_col=["cln enc id"],
)

latest_bmi_df = pd.merge(bmi_df, enc_base_df, on="cln enc id").sort_values(["cln enc date"]).drop_duplicates(subset="enterpriseid", keep="last")[["enterpriseid", "enc BMI"]]

display(
    latest_bmi_df[latest_bmi_df["enc BMI"] >=35]["enterpriseid"]
    )


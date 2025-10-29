#%%
import pandas as pd
from  pathlib import Path
from datetime import datetime, timedelta
import os
import re

today = pd.Timestamp.today().normalize()

ten_years_ago = today - pd.DateOffset(years=10)
five_years_ago = today - pd.DateOffset(years=5)
three_years_ago = today - pd.DateOffset(years=3)
two_years_ago = today - pd.DateOffset(years=2)
one_year_ago = today - pd.DateOffset(years=1)

def get_latest_data_file(file_string: str, data_dir: str | Path = "data")->str:
    """Gets requested data file from "data" directory 

    file_string takes string of requested data file

    Returns directory last alphabetical file name containing file_string"""
    
    base = Path(data_dir)

    if not base.is_absolute():
        base = Path.cwd() / base
    if not base.exists():
        raise FileNotFoundError(f"Data Directory not found: {base}")
    
    dir_path = os.path.join(os.getcwd(),"data")
    file_list = [
        p for p in base.iterdir() 
        if p.is_file()
        and file_string.lower() in p.name.lower()
    ]

    if not file_list:
        raise FileNotFoundError(
            f'No files in "{base}" containing "{file_string}"'
        )

    file_list.sort(key = lambda p: p.name.lower(), reverse=True)
    return os.path.join(dir_path,file_list[0])

# Open patient file as dataframe
demographics_df = pd.read_csv(get_latest_data_file("demographics"), index_col="enterpriseid")



#%% ---------------------------WELLNESS VISITS-----------------------------

# Open encounter data file as dataframe
enc_base_df = pd.read_csv(get_latest_data_file("encounter base"), 
                          index_col="cln enc id",
                          parse_dates=["cln enc date"])

# Filter appointments by wellness visits occuring on or after 1/1/2025
pattern = "|".join(["well", "awv"])

enc_w_awv_df = enc_base_df.loc[
        (enc_base_df["appttype"].str.contains(pattern, case=False, na=False))
        & (enc_base_df["cln enc date"] >= "2025-01-01")
        ]

ids_index = pd.Index(enc_w_awv_df["enterpriseid"].dropna().astype(demographics_df.index.dtype))

# Find all patients without a wellness visit since 1/1/2025
pt_wo_awv_df = demographics_df.loc[(demographics_df.index.difference(ids_index))]

ov65_wo_awv = pt_wo_awv_df.loc[(pt_wo_awv_df["age"] >= 40)]

ov65_wo_awv.head()



#%% --------------------------COLON CANCER SCREENING----------------------------

# Open surgical history file as dataframe
surghx_df = pd.read_csv(get_latest_data_file("surgical history"),
                        parse_dates=["surg hist date"])

# Open cologuard file as dataframe
cologuard_df = pd.read_csv(get_latest_data_file("cologuard"),
                           parse_dates=["labdate"])

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
cologua_df = cologuard_df.loc[
    (cologuard_df["labdate"] >= three_years_ago)
]

ids_colonos = pd.Index(colonos_df["enterpriseid"].dropna().unique())
ids_flexsig = pd.Index(flexsig_df["enterpriseid"].dropna().unique())
ids_cologua = pd.Index(cologua_df["enterpriseid"].dropna().unique())

colo_complete = ids_colonos.union(ids_flexsig).union(ids_cologua)

pt_wo_colo_df = demographics_df[
    ~demographics_df.index.isin(colo_complete)
    & (demographics_df["age"] >= 45)]


pt_wo_colo_df.head()

# Generate a dataframe of all patients with completed colon cancer screening modalities 
# colo_complete_df = pd.concat([
#     colonos_df["enterpriseid"],
#     flexsig_df["enterpriseid"], 
#     cologua_df["enterpriseid"]
#     ]).drop_duplicates().reset_index(drop=True)


# patient_df.join(encounter_df, on="enterpriseid")[[]]

#%% ---------------------------BREAST CANCER SCREENING------------------------------

mammogram_df = pd.read_csv(get_latest_data_file("mammogram"), parse_dates=["dt f lst mmmgrm"])
mammogram_df["dt f lst mmmgrm"] = pd.to_datetime(mammogram_df["dt f lst mmmgrm"], errors="coerce")

mammo_df = mammogram_df[mammogram_df["dt f lst mmmgrm"] >= two_years_ago]

pt_wo_mammo = demographics_df[
    (~demographics_df.index.isin(mammo_df["enterpriseid"])) 
    & (demographics_df["age"] >= 40)
    & (demographics_df["patientsex"] == "F")
]

pt_wo_mammo.head()

#%% -------------------------DIABETIC SCREENING-----------------------

a1c_df = pd.read_csv(get_latest_data_file("a1c"), parse_dates=["labdate"])

screen_a1c_df = a1c_df[a1c_df["labdate"] >= two_years_ago]

pt_wo_a1c = demographics_df[
    (~demographics_df.index.isin(screen_a1c_df["enterpriseid"]))
    & (demographics_df["age"] >= 40)
    ]

pt_wo_a1c.head() 

#%% -----------------------A1C OVER 9------------------------------

a1c_df = pd.read_csv(get_latest_data_file("a1c"), parse_dates=["labdate"])
# a1c_df["labvalue"] = a1c_df["labvalue"].astype(str)


# print(sorted(a1c_df["labvalue"].unique()))

a1c_over_9_df = a1c_df[
    (a1c_df["labdate"] >= two_years_ago)
    & (a1c_df["labvalue"] >= 9)
    ]

pt_w_uncont_dm = demographics_df[demographics_df.index.isin(a1c_over_9_df["enterpriseid"])]

pt_w_uncont_dm.head()

#%%

a1c_df = pd.read_csv(get_latest_data_file("a1c"), parse_dates=["labdate"])



a1c_df["labvalue"].unique()

# %%

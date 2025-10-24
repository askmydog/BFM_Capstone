#%%
import pandas as pd
from  pathlib import Path
from datetime import datetime, timedelta
import os

today = pd.Timestamp.today().normalize()

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
patient_df = pd.read_csv(get_latest_data_file("demographics"), index_col="enterpriseid")

# Convert DOB field to datetime object
patient_df["patientdob"] = pd.to_datetime(patient_df["patientdob"])

# Calculate patient age from today, first create dataframe that will calculate if it is before the patient's birthday
before_birthday = (today.month < patient_df["patientdob"].dt.month | 
                   ((today.month == patient_df["patientdob"].dt.month) & (today.day < patient_df["patientdob"].dt.month)))

patient_df["patient age"] = (today.year - patient_df["patientdob"].dt.year - before_birthday)

# Remove deceased and inactive patients
patient_df = patient_df[
    (patient_df["status"] == "a") 
    & (patient_df["ptnt dcsd ysn"].isna())]

# Open encounter data file as dataframe
enc_base_df = pd.read_csv(get_latest_data_file("encounter base"), index_col="cln enc id")

# Convert clinical encounter date to datetime object
enc_base_df["cln enc date"] = pd.to_datetime(enc_base_df["cln enc date"])

# Open surgical history file as dataframe
surghx_df = pd.read_csv(get_latest_data_file("surgical history"))

# Convert surgical history date to datetime object
surghx_df["surg hist date"] = pd.to_datetime(surghx_df["surg hist date"])

# Open cologuard file as dataframe
cologuard_df = pd.read_csv(get_latest_data_file("cologuard"))

# Convert cologuard date to datetime object
cologuard_df["labdate"] = pd.to_datetime(cologuard_df["labdate"])


#%% ---------------------------WELLNESS VISITS-----------------------------
# Filter appointments by wellness visits occuring on or after 1/1/2025

pattern = "|".join(["well", "awv"])

enc_w_awv_df = enc_base_df.loc[
        (enc_base_df["appttype"].str.contains(pattern, case=False, na=False))
        & (enc_base_df["cln enc date"] >= "2025-01-01")
        ]

# Find all patients without a wellness visit since 1/1/2025
pt_wo_awv_df = patient_df.loc[(patient_df.index.difference(enc_w_awv_df["enterpriseid"]))]



ov65_wo_awv = pt_wo_awv_df.loc[(pt_wo_awv_df["patient age"] >= 65)]

ov65_wo_awv.head()

#%% --------------------------COLON CANCER SCREENING----------------------------

today = pd.Timestamp.today().normalize()

ten_years_ago = today - pd.DateOffset(years=10)
five_years_ago = today - pd.DateOffset(years=5)
three_years_ago = today - pd.DateOffset(years=3)


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

pt_wo_colo_df = patient_df[
    ~patient_df.index.isin(colo_complete)
    & (patient_df["patient age"] >= 45)]


pt_wo_colo_df

# Generate a dataframe of all patients with completed colon cancer screening modalities 
# colo_complete_df = pd.concat([
#     colonos_df["enterpriseid"],
#     flexsig_df["enterpriseid"], 
#     cologua_df["enterpriseid"]
#     ]).drop_duplicates().reset_index(drop=True)


# patient_df.join(encounter_df, on="enterpriseid")[[]]





#%%

patient_df["ptnt dcsd ysn"].isna()
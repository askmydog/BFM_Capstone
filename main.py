#%%
import pandas as pd
from  pathlib import Path
from datetime import datetime, timedelta
import os
import re
from IPython.display import display

today = pd.Timestamp.today().normalize()

ten_years_ago              = today - pd.DateOffset(years=10)
five_years_ago             = today - pd.DateOffset(years=5)
three_years_ago            = today - pd.DateOffset(years=3)
two_years_ago              = today - pd.DateOffset(years=2)
one_year_ago               = today - pd.DateOffset(years=1)
beginning_of_cal_year      = pd.Timestamp(today.year, 1, 1)
beginning_of_last_cal_year = pd.Timestamp(today.year - 1, 1, 1)


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
        & (enc_base_df["cln enc date"] >= beginning_of_cal_year)
        ]


ids_index = pd.Index(enc_w_awv_df["enterpriseid"].dropna().astype(demographics_df.index.dtype))

# Find all patients without a wellness visit since 1/1/2025
pt_wo_awv_df = demographics_df.loc[(demographics_df.index.difference(ids_index))]

ov40_wo_awv = pt_wo_awv_df.loc[(pt_wo_awv_df["age"] >= 40)]

ov40_wo_awv.head()



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

recent_mammo_df = mammogram_df[mammogram_df["dt f lst mmmgrm"] >= two_years_ago]

pt_wo_mammo_df = demographics_df[
    (~demographics_df.index.isin(recent_mammo_df["enterpriseid"])) 
    & (demographics_df["age"] >= 40)
    & (demographics_df["patientsex"] == "F")
]

pt_wo_mammo_df.head()

mammogram_df.dropna(subset=["mmmgrm rslt"], inplace=True)

pt_w_abnml_mammo_df = mammogram_df[mammogram_df["mmmgrm rslt"].dropna().str.contains("Abnormal")]
display(pt_w_abnml_mammo_df)

#%% -------------------------DIABETIC SCREENING-----------------------

a1c_df = pd.read_csv(get_latest_data_file("a1c"), 
                     parse_dates=["labdate"],
                     dtype={"labvalue":float})

recent_a1c_df = a1c_df[a1c_df["labdate"] >= two_years_ago]

pt_wo_a1c = demographics_df[
    (~demographics_df.index.isin(recent_a1c_df["enterpriseid"]))
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

#%% --------------------- ABNORMAL VITALS (BP AND BMI)--------------------------

vitals_df = pd.read_csv(get_latest_data_file("vitals"))
enc_base_df = pd.read_csv(get_latest_data_file("encounter base"), 
                          parse_dates=["cln enc date"], 
                          index_col=["cln enc id"])

latest_enc_df = (
    enc_base_df
    .sort_values(["enterpriseid", "cln enc date"], ascending=[True, False])
    .drop_duplicates(subset="enterpriseid", keep="first")
)

latest_vitals_df = pd.merge(vitals_df, latest_enc_df, on="cln enc id").sort_values(["cln enc date"])

pt_w_sbp_ov_140_df = latest_vitals_df[(latest_vitals_df["sys BP"] >= 140) & (latest_vitals_df["cln enc date"] >= one_year_ago)]
pt_w_dbp_ov_90_df = latest_vitals_df[(latest_vitals_df["dia BP"] >= 90) & (latest_vitals_df["cln enc date"] >= one_year_ago)]
bmi_ov_35_df = latest_vitals_df[(latest_vitals_df["enc BMI"] >= 35) & (latest_vitals_df["cln enc date"] >= one_year_ago)]


# print(vitals_df["cln enc id"].head())
# print(enc_base_df["cln enc id"].head())
display(bmi_ov_35_df)


#%% ------------------------- DIAGNOSIS GROUPS ------------------------------


def get_diagnosis_dataframe(dx_group: str, 
                            enc_base_df: pd.DataFrame, 
                            enc_dx_df: pd.DataFrame,
                            time_window: pd.Timestamp,
                            num_diagnoses: int
                            ) -> pd.DataFrame:
    """
    Returns dataframe of diagnosis group of interest, with diagnosis group column 
    that represents whether the diagnosis was made on at least the number of specified times 
    within the specified time window. 
    """
    
    if dx_group not in enc_dx_df.columns:
        raise ValueError(f"{dx_group} not in enc_dx_df list of columns: {list(enc_dx_df.columns)}")
    
    # Groupby encounter diagnosis dataframe by clinical encounter ID and assign a 1 to the target column if a diagnosis code is present for that encounter
    dx_by_enc_df = enc_dx_df.groupby("cln enc id")[dx_group].max()

    # Merge dataframe with enc_base_df to get patient ID and date data
    enc_dx_merged_df = pd.merge(enc_base_df, dx_by_enc_df, left_index=True, right_on="cln enc id")

    # Filter clinical encounters occuring within the time window of interest
    recent_enc_dx_df = enc_dx_merged_df[enc_dx_merged_df["cln enc date"] >= time_window]

    # Groupby recent clin enc dataframe by enterpriseid and sum diagnosis group of interest
    sum_dx_by_entid = recent_enc_dx_df.groupby("enterpriseid")[dx_group].sum().rename(dx_group)

    # return dataframe with dx_group column containing 1 if gte specified number of diagnoses in the time window of interest
    return (sum_dx_by_entid >= num_diagnoses).astype("Int8").to_frame()


enc_base_df = pd.read_csv(get_latest_data_file("encounter base"), 
                          parse_dates=["cln enc date"], 
                          index_col=["cln enc id"])
enc_dx_df = pd.read_csv(get_latest_data_file("encounter diagnoses"))
two_years_ago = today - pd.DateOffset(years=2)
num_diagnoses = 2

pt_w_dm_df       = get_diagnosis_dataframe("dm",       enc_base_df, enc_dx_df, two_years_ago, num_diagnoses)
pt_w_ascvd_df    = get_diagnosis_dataframe("ascvd",    enc_base_df, enc_dx_df, two_years_ago, num_diagnoses) 
pt_w_htn_df      = get_diagnosis_dataframe("htn",      enc_base_df, enc_dx_df, two_years_ago, num_diagnoses)
pt_w_hlp_df      = get_diagnosis_dataframe("hlp",      enc_base_df, enc_dx_df, two_years_ago, num_diagnoses)
pt_w_mood_df     = get_diagnosis_dataframe("mood",     enc_base_df, enc_dx_df, two_years_ago, num_diagnoses)
pt_w_learndis_df = get_diagnosis_dataframe("learndis", enc_base_df, enc_dx_df, two_years_ago, num_diagnoses)
pt_w_dementia_df = get_diagnosis_dataframe("dementia", enc_base_df, enc_dx_df, two_years_ago, num_diagnoses)
pt_w_sud_df      = get_diagnosis_dataframe("sud",      enc_base_df, enc_dx_df, two_years_ago, num_diagnoses)
pt_w_sdoh_df     = get_diagnosis_dataframe("sdoh",     enc_base_df, enc_dx_df, two_years_ago, num_diagnoses)
pt_w_noncomp_df  = get_diagnosis_dataframe("noncomp",  enc_base_df, enc_dx_df, two_years_ago, num_diagnoses)



pt_dx_reconstructed = (pt_w_dm_df
                       .merge(pt_w_ascvd_df, left_index=True, right_index=True)
                       .merge(pt_w_htn_df, left_index=True, right_index=True)
                       .merge(pt_w_hlp_df, left_index=True, right_index=True)
                       .merge(pt_w_mood_df, left_index=True, right_index=True)
                       .merge(pt_w_learndis_df, left_index=True, right_index=True)
                       .merge(pt_w_dementia_df, left_index=True, right_index=True)
                       .merge(pt_w_sud_df, left_index=True, right_index=True)
                       .merge(pt_w_sdoh_df, left_index=True, right_index=True)
                       .merge(pt_w_noncomp_df, left_index=True, right_index=True)
                       )

display(pt_dx_reconstructed)

#%%--------------------------- KIDNEY EVALUTION IN DIABETES (KED) ---------------------



#%% ------------------------------HIGH RAF SCORE-------------------------------

raf_df = pd.read_csv(get_latest_data_file("raf"), 
                     index_col=["enterpriseid"],
                     dtype={"RAF score": float})
9
pt_w_raf_ov_1 = raf_df[raf_df["RAF score"] >= 1]

display( pt_w_raf_ov_1)

#%% ---------------------------------Medications------------------------------



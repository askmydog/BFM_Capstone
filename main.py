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


#%% ----------------------- SEX ----------------------------------

# patient_sex = demographics_df["patientsex"].copy().to_frame()

s = demographics_df["patientsex"].astype("str")

patient_sex = pd.DataFrame({
    "is_female":   (s == "F").astype("Int8"),
    "is_male":     (s == "M").astype("Int8"),
    "sex_unknown": (~(s == "F") & ~(s == "M")).astype("int8")
    },
    index = demographics_df.index
    )

display(patient_sex)

#%% --------------------------- AGE -----------------------------------

a = demographics_df["age"].astype("int16")

patient_age = pd.DataFrame({
    "<18": (a < 18).astype("int8"),
    "18-29": ((a >= 18) & (a < 30)).astype("int8"),
    "30-39": ((a >= 30) & (a < 40)).astype("int8"),
    "40-49": ((a >= 40) & (a < 50)).astype("int8"),
    "50-59": ((a >= 50) & (a < 60)).astype("int8"),
    "60-69": ((a >= 60) & (a < 70)).astype("int8"),
    "70-79": ((a >= 70) & (a < 80)).astype("int8"),
    "80-89": ((a >= 80) & (a < 90)).astype("int8"),
    ">=90": (a >= 90).astype("int8"),
})

display(patient_age)



#%% ------------------------ 




#%% -------------------------- COLON CANCER SCREENING ----------------------------

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

#%% ---------------------------- DIAGNOSIS GROUPS, REFACTORED  ---------------------------------
enc_base_df = pd.read_csv(get_latest_data_file("encounter base"), 
                          parse_dates=["cln enc date"], 
                          index_col=["cln enc id"])
enc_dx_df = pd.read_csv(get_latest_data_file("encounter diagnoses"))

dx_groups = ["dm","ascvd","htn","hlp","mood","learndis","dementia","sud","sdoh","noncomp"]

# Clean dx table
dx = enc_dx_df.copy()
dx[dx_groups] = (dx[dx_groups]
                 .apply(pd.to_numeric, errors="coerce")
                 .fillna(0).clip(0,1).astype("Int8"))

# One row per encounter with flags (any dx in that encounter)
dx_by_enc = dx.groupby("cln enc id", as_index=True)[dx_groups].max()

# Join to encounters
enc = enc_base_df.copy()
enc["cln enc date"] = pd.to_datetime(enc["cln enc date"], errors="coerce")
if enc.index.name != "cln enc id":
    enc = enc.set_index("cln enc id", drop=False)

enc_with = enc.join(dx_by_enc, how="left").fillna(0)

# Filter window
recent = enc_with.loc[enc_with["cln enc date"] >= two_years_ago]

# Counts and flags per patient
pt_counts = recent.groupby("enterpriseid", as_index=True)[dx_groups].sum()
pt_diagnoses  = (pt_counts >= 2 ).astype("Int8")


display(pt_diagnoses)

#%% -------------------------- KED --------------------------------------

ked_df = pd.read_csv("data/ked.csv", parse_dates=["labdate"])

recent_ked_df = ked_df.loc[ked_df["labdate"] >= beginning_of_cal_year].copy()

recent_ked_df["gfr"]  = (recent_ked_df["labanalyte"] == "GFR").astype("Int8")
recent_ked_df["uacr"] = (recent_ked_df["labanalyte"] == "UACR").astype("Int8")

ked_by_entid_df = recent_ked_df.groupby("enterpriseid")[["uacr", "gfr"]].max()

has_recent_ked = ked_by_entid_df[(ked_by_entid_df["uacr"] == 1) & (ked_by_entid_df["gfr"] == 1)]

diab_pts = pt_diagnoses[pt_diagnoses["dm"]==1]

diab_wo_ked = diab_pts[~diab_pts.index.isin(has_recent_ked.index)]

display(diab_wo_ked)

#%% --------------------- TOBACCO USE --------------------------------------------

tobacco_use = demographics_df.copy()

# display(tobacco_use.columns)

def classify_tobacco_use(s: str) -> str:
    if not isinstance(s, str):
        return "unknown"
    s = s.lower()
    if "never" in s:
        return "never"
    if "former" in s:
        return "former"
    if "current" in s:
        return "current"
    else:
        return "unknown" 

tobacco_use["tob_classifier"] = tobacco_use["tobacco use"].apply(classify_tobacco_use)

tobacco_use["never"]   = (tobacco_use["tob_classifier"] == "never").astype("Int8")
tobacco_use["former"]  = (tobacco_use["tob_classifier"] == "former").astype("Int8")
tobacco_use["current"] = (tobacco_use["tob_classifier"] == "current").astype("Int8")
tobacco_use["unknown"] = (tobacco_use["tob_classifier"] == "unknown").astype("Int8")


display(tobacco_use)


#%% ------------------------- INSURANCE ------------------------------------------

pd.option_context("display.max_rows", None)
display(demographics_df["patient primary ins pkg name"].value_counts().head(30))



def classify_insurance(ins:str) -> str:
    if re.search("self pay", ins, re.I):
        return "None"
    if re.search("medicare", ins, re.I):
        return "Medicare"
    if re.search("tricare|champva|fep", ins, re.I):
        return "Mil/Fed"
    if re.search("medicaid", ins, re.I):
        return "Medicaid"
    else:
        return "Commercial"

insurance = pd.DataFrame(index = demographics_df.index)


insurance["prim"] = demographics_df["patient primary ins pkg name"].apply(classify_insurance)
insurance["sec"]  = demographics_df["patient secondary ins pkg name"].apply(classify_insurance)


insurance["none"]       = ((insurance["prim"] == "None")       & (insurance["sec"] == "None")).astype("int8")
insurance["medicare"]   = ((insurance["prim"] == "Medicare")   | (insurance["sec"] == "Medicare")).astype("int8")
insurance["mil/fed"]    = ((insurance["prim"] == "Mil/Fed")    | (insurance["sec"] == "Mil/Fed")).astype("int8")
insurance["medicaid"]   = ((insurance["prim"] == "Medicaid")   | (insurance["sec"] == "Medicaid")).astype("int8")
insurance["commercial"] = ((insurance["prim"] == "Commercial") | (insurance["sec"] == "Commercial")).astype("int8")


display(insurance.head(100))
    


#%% -------------------------- MEDICATIONS -------------------------------------------

pt_meds = pd.read_csv("data/med list.csv").groupby("enterpriseid").max()
pt_meds.drop(columns=["med name"], inplace=True)

display(pt_meds)

#%%--------------------------- KIDNEY EVALUTION IN DIABETES (KED) ---------------------



#%% ------------------------------HIGH RAF SCORE-------------------------------

raf_df = pd.read_csv(get_latest_data_file("raf"), 
                     index_col=["enterpriseid"],
                     dtype={"RAF score": float})

pt_w_raf_ov_1 = raf_df[raf_df["RAF score"] >= 1]

display( pt_w_raf_ov_1)

#%% ---------------------------------Medications------------------------------



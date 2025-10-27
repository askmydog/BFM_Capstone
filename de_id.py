# #%%
# import pandas as pd
# from  pathlib import Path
# import os
# import numpy as np

# input_dir = "capstone/data"
# data_targets = [
#     "demographics",
#     "encounter base",
#     "encounter diagnoses",
#     "cologuard",
#     "surgical history",
#     "a1c",
#     "ked",
#     "mammogram",
#     "med list",
# ]

# def get_data_files(input_dir: str = input_dir, data_targets: list = data_targets) -> dict:
#     """
#     Gets list of file names
#     """
#     file_path_dict = dict()

#     input_base = Path(input_dir)
    
#     if not input_base.is_absolute():
#         input_base = Path.cwd().parent / input_base
#     if not input_base.exists():
#         raise FileNotFoundError(f"No directory at location {input_base}")

#     for target in data_targets:
#         file_list = list()
#         file_list = [f for f in input_base.iterdir()
#                      if f.is_file() and
#                      target.lower() in f.name.lower()]
#         if len(file_list) == 0:
#             raise FileNotFoundError(f"No file matching {target} in {input_base}")
#         file_list.sort(reverse=True)
#         file_path_dict[target] = file_list[0]
    
#     return file_path_dict
#     # target for target in data_targets:


# # Declare empty dataframes of data files of interest
# demographics_df = pd.DataFrame()
# enc_base_df = pd.DataFrame()
# enc_dx_df = pd.DataFrame()
# colog_df = pd.DataFrame()
# surg_hx_df = pd.DataFrame()
# a1c_df = pd.DataFrame()
# ked_df = pd.DataFrame()
# mammo_df = pd.DataFrame()
# med_list_df = pd.DataFrame()


# for name, file in get_data_files().items():
#     if name == "demographics": 
#         demographics_df = pd.read_csv(file)
#         demographics_df["patientdob"] = pd.to_datetime(demographics_df["patientdob"])
#     if name == "encounter base": 
#         enc_base_df = pd.read_csv(file)
#         enc_base_df["cln enc date"] = pd.to_datetime(enc_base_df["cln enc date"])
#     if name == "encounter diagnoses": enc_dx_df = pd.read_csv(file)
#     if name == "cologuard": 
#         colog_df = pd.read_csv(file)
#         colog_df["labdate"] = pd.to_datetime(colog_df["labdate"])
#     if name == "surgical history": surg_hx_df = pd.read_csv(file)
#     if name == "a1c": a1c_df = pd.read_csv(file)
#     if name == "ked": ked_df = pd.read_csv(file)
#     if name == "mammogram": mammo_df = pd.read_csv(file)
#     if name == "med list": med_list_df = pd.read_csv(file)

# #%% -------------------------Calcuate ages from DOB--------------------------

# today = pd.Timestamp.today().normalize()

# def calc_age(dob: pd.Timestamp, ref_day: pd.Timestamp = today):
#     years = ref_day.year - dob.year
#     if dob.month < ref_day.month:
#         return years - 1
#     elif (dob.month == ref_day.month) & (dob.day < ref_day.day):
#         return years - 1
#     else:
#         return years
    
# demographics_df["age"] = demographics_df["patientdob"].apply(calc_age)

# display(demographics_df.head())

# #%% -------------------------Assign new random IDs------------------------------

# value_pool = list(range(100_000, 1_000_000))
# num_rows = len(demographics_df)
# rand_uniq_values = np.random.choice(value_pool, num_rows, replace=False)

# id_key = pd.DataFrame(rand_uniq_values, index=demographics_df["enterpriseid"], columns=["rand_id"])
# # demographics_df["rand_id"] = rand_uniq_values
    
# id_key.head()

# def assign_new_id(old_id: int):
#     if id_key.index.isin([old_id]).any():
#         return id_key.loc[old_id, "rand_id"]

# demographics_df["enterpriseid"] = demographics_df["enterpriseid"].apply(assign_new_id)
# enc_base_df["enterpriseid"] = enc_base_df["enterpriseid"].apply(assign_new_id)

# enc_base_df.head()

# # print(dfs)
# # masked_df = pd.read_csv()

#%%
from pathlib import Path
import re
import numpy as np
import pandas as pd

INPUT_DIR = Path("capstone/data")
DATA_TARGETS = {
    "demographics":      ("demographics",      ("csv",)),
    "encounter base":    ("encounter base",    ("csv",)),
    "encounter diagnoses": ("encounter diagnoses", ("csv",)),
    "cologuard":         ("cologuard",         ("csv",)),
    "surgical history":  ("surgical history",  ("csv",)),
    "a1c":               ("a1c",               ("csv",)),
    "ked":               ("ked",               ("csv",)),
    "mammogram":         ("mammogram",         ("csv",)),
    "med list":          ("med list",          ("csv",)),
}

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.casefold() for t in re.split(r'(\d+)', s)]

def get_latest_files(input_dir: Path = INPUT_DIR, targets: dict = DATA_TARGETS) -> dict:
    """Return the newest-matching file per target (by natural filename sort)."""
    base = input_dir if input_dir.is_absolute() else Path.cwd().parent / input_dir
    if not base.exists():
        raise FileNotFoundError(f"No directory at {base}")
    out: dict = {}
    for key, (needle, exts) in targets.items():
        exts = {("." + e.lower().lstrip(".")) for e in exts}
        matches = [p for p in base.iterdir()
                   if p.is_file()
                   and p.suffix.lower() in exts
                   and needle.casefold() in p.name.casefold()]
        if not matches:
            raise FileNotFoundError(f"No file matching '{needle}' with {sorted(exts)} in {base}")
        matches.sort(key=lambda p: _natural_key(p.name), reverse=True)
        out[key] = matches[0]
    return out

# ---- Read CSVs (parse dates up front)
files = get_latest_files()

demographics_df = pd.read_csv(files["demographics"], parse_dates=["patientdob"])
demographics_df["patientsex"].fillna(value='U', inplace=True)
enc_base_df     = pd.read_csv(files["encounter base"], parse_dates=["cln enc date"])
enc_dx_df       = pd.read_csv(files["encounter diagnoses"])
colog_df        = pd.read_csv(files["cologuard"], parse_dates=["labdate"])
surg_hx_df      = pd.read_csv(files["surgical history"], parse_dates=["surg hist date"])
a1c_df          = pd.read_csv(files["a1c"], parse_dates=["labdate"])
ked_df          = pd.read_csv(files["ked"], parse_dates=["labdate"])
mammo_df        = pd.read_csv(files["mammogram"], parse_dates=["dt f lst mmmgrm"])
med_list_df     = pd.read_csv(files["med list"])

# --------------- Clean up Demographics dataframe Calculate ages (vectorized, correct), anonymize zip and remove inactive and desceased patients -------------------------
today = pd.Timestamp.today().normalize()

# Ensure datetime
# demographics_df["patientdob"] = pd.to_datetime(demographics_df["patientdob"], errors="coerce")

before_birthday = (
    (today.month < demographics_df["patientdob"].dt.month) |
    ((today.month == demographics_df["patientdob"].dt.month) &
     (today.day   < demographics_df["patientdob"].dt.day))
)

demographics_df["age"] = (
    today.year - demographics_df["patientdob"].dt.year - before_birthday.astype(int)
)


# def anonymize_zipcode(zipcode:str):

# # Remove all but first 3 digits of zip code

# first3 = demographics_df["patient zip"].astype("string").str.extract(r'^\s*(\d{3})', expand=False)

# demographics_df["patient zip"] = demographics_df["patient zip"].fillna('234').astype(int).mul(100)

demographics_df["patient zip"] = demographics_df["patient zip"].apply(lambda zipcode: 
                                                                      int(str(zipcode).strip().split("-")[0][:3]+"00") 
                                                                      if (zipcode != np.nan) 
                                                                      and (str(zipcode)[0].isdigit()) 
                                                                      else 23400)

# print(demographics_df["patient zip"].unique())

# Remove inactive and deceased patients
demographics_df = demographics_df[
    (demographics_df["status"] == "a") 
    & (demographics_df["ptnt dcsd ysn"].isna())]

# demographics_df.head()

del demographics_df["patientdob"]
del demographics_df["status"]
del demographics_df["ptnt dcsd ysn"]


# ------------------------- Assign new random IDs (fast, reproducible) -------------------
# 1) Unique enterprise IDs
ent = demographics_df["enterpriseid"].astype("Int64")  # align dtype; change if yours is str
uids = ent.dropna().unique()

# 2) Draw unique 6-digit numbers
rng = np.random.default_rng(42)                         # set seed for reproducibility
rand_pool = np.arange(100_000, 1_000_000, dtype=np.int64)
rand_ids  = rng.choice(rand_pool, size=len(uids), replace=False)

# 3) Build mapping Series and (optionally) a two-column key DF to persist
id_map   = pd.Series(rand_ids, index=uids, name="rand_id")
id_key   = id_map.rename_axis("enterpriseid").reset_index()

# 4) Map onto dataframes (no apply)
demographics_df["enterpriseid"] = demographics_df["enterpriseid"].map(id_map)
enc_base_df["enterpriseid"]     = enc_base_df["enterpriseid"].map(id_map)
colog_df["enterpriseid"]        = colog_df["enterpriseid"].map(id_map)
surg_hx_df["enterpriseid"]      = surg_hx_df["enterpriseid"].map(id_map)
a1c_df["enterpriseid"]          = a1c_df["enterpriseid"].map(id_map)
ked_df["enterpriseid"]          = ked_df["enterpriseid"].map(id_map)
mammo_df["enterpriseid"]        = mammo_df["enterpriseid"].map(id_map)
med_list_df["enterpriseid"]     = med_list_df["enterpriseid"].map(id_map)

# display(demographics_df[demographics_df.duplicated(subset=["enterpriseid"])])

# # Sanity checks
# assert demographics_df["rand_id"].notna().all(), "Some enterprise IDs had no mapping."
# assert demographics_df["rand_id"].is_unique, "Random IDs are not unique."

# display(demographics_df.head(), enc_base_df.head(), id_key.head())
# ------------------Copy each dataframe to a corresponding csv file----------------

output_dir = Path.cwd() / Path("data")


demographics_df.to_csv(output_dir / "demographics.csv", index=False)
enc_base_df.to_csv(output_dir / "encounter base.csv", index=False)
enc_dx_df.to_csv(output_dir / "encounter diagnoses.csv", index=False)
colog_df.to_csv(output_dir / "cologuard.csv", index=False)
surg_hx_df.to_csv(output_dir / "surgical history.csv", index=False)
a1c_df.to_csv(output_dir / "a1c.csv", index=False)
ked_df.to_csv(output_dir / "ked.csv", index=False)
mammo_df.to_csv(output_dir / "mammogram.csv", index=False)
med_list_df.to_csv(output_dir / "med list.csv", index=False)

colog_df["enterpriseid"]        = colog_df["enterpriseid"].map(id_map)
surg_hx_df["enterpriseid"]      = surg_hx_df["enterpriseid"].map(id_map)
a1c_df["enterpriseid"]          = a1c_df["enterpriseid"].map(id_map)
ked_df["enterpriseid"]          = ked_df["enterpriseid"].map(id_map)
mammo_df["enterpriseid"]        = mammo_df["enterpriseid"].map(id_map)
med_list_df["enterpriseid"]     = med_list_df["enterpriseid"].map(id_map)


# %%

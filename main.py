# %%
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


# %% ---------------------------WELLNESS VISITS-----------------------------

# Open encounter data file as dataframe
enc_base_df = pd.read_csv(
    get_latest_data_file("encounter base"),
    index_col="cln enc id",
    parse_dates=["cln enc date"],
)

# Filter appointments by wellness visits occuring on or after 1/1/2025
pattern = "|".join(["well", "awv"])

enc_w_awv_df = enc_base_df.loc[
    (enc_base_df["appttype"].str.contains(pattern, case=False, na=False))
    & (enc_base_df["cln enc date"] >= beginning_of_cal_year)
]


ids_index = pd.Index(
    enc_w_awv_df["enterpriseid"].dropna().astype(demographics_df.index.dtype)
)

# Find all patients without a wellness visit since 1/1/2025
pt_wo_awv_df = demographics_df.loc[(demographics_df.index.difference(ids_index))]

ov40_wo_awv = pt_wo_awv_df.loc[(pt_wo_awv_df["age"] >= 40)]

ov40_wo_awv.head()


# %% --------------------------COLON CANCER SCREENING----------------------------

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

pt_wo_colo_df = demographics_df[
    ~demographics_df.index.isin(colo_complete) & (demographics_df["age"] >= 45)
]


pt_wo_colo_df.head()

# Generate a dataframe of all patients with completed colon cancer screening modalities
# colo_complete_df = pd.concat([
#     colonos_df["enterpriseid"],
#     flexsig_df["enterpriseid"],
#     cologua_df["enterpriseid"]
#     ]).drop_duplicates().reset_index(drop=True)


# patient_df.join(encounter_df, on="enterpriseid")[[]]

# %% ---------------------------BREAST CANCER SCREENING------------------------------

mammogram_df = pd.read_csv(
    get_latest_data_file("mammogram"), parse_dates=["dt f lst mmmgrm"]
)
mammogram_df["dt f lst mmmgrm"] = pd.to_datetime(
    mammogram_df["dt f lst mmmgrm"], errors="coerce"
)

recent_mammo_df = mammogram_df[mammogram_df["dt f lst mmmgrm"] >= two_years_ago]

pt_wo_mammo_df = demographics_df[
    (~demographics_df.index.isin(recent_mammo_df["enterpriseid"]))
    & (demographics_df["age"] >= 40)
    & (demographics_df["patientsex"] == "F")
]

pt_wo_mammo_df.head()

mammogram_df.dropna(subset=["mmmgrm rslt"], inplace=True)

pt_w_abnml_mammo_df = mammogram_df[
    mammogram_df["mmmgrm rslt"].dropna().str.contains("Abnormal")
]
# display(pt_w_abnml_mammo_df)

# %% -------------------------DIABETIC SCREENING-----------------------

a1c_df = pd.read_csv(
    get_latest_data_file("a1c"), parse_dates=["labdate"], dtype={"labvalue": float}
)

recent_a1c_df = a1c_df[a1c_df["labdate"] >= two_years_ago]

pt_wo_a1c = demographics_df[
    (~demographics_df.index.isin(recent_a1c_df["enterpriseid"]))
    & (demographics_df["age"] >= 40)
]

pt_wo_a1c.head()

# %% -----------------------A1C OVER 9------------------------------

a1c_df = pd.read_csv(get_latest_data_file("a1c"), parse_dates=["labdate"])
# a1c_df["labvalue"] = a1c_df["labvalue"].astype(str)


# print(sorted(a1c_df["labvalue"].unique()))

a1c_over_9_df = a1c_df[(a1c_df["labdate"] >= two_years_ago) & (a1c_df["labvalue"] >= 9)]

pt_w_uncont_dm = demographics_df[
    demographics_df.index.isin(a1c_over_9_df["enterpriseid"])
]

pt_w_uncont_dm.head()

# %% --------------------- ABNORMAL VITALS (BP AND BMI)--------------------------

vitals_df = pd.read_csv(get_latest_data_file("vitals"))
enc_base_df = pd.read_csv(
    get_latest_data_file("encounter base"),
    parse_dates=["cln enc date"],
    index_col=["cln enc id"],
)

latest_enc_df = enc_base_df.sort_values(
    ["enterpriseid", "cln enc date"], ascending=[True, False]
).drop_duplicates(subset="enterpriseid", keep="first")

latest_vitals_df = pd.merge(vitals_df, latest_enc_df, on="cln enc id").sort_values(
    ["cln enc date"]
)

pt_w_sbp_ov_140_df = latest_vitals_df[
    (latest_vitals_df["sys BP"] >= 140)
    & (latest_vitals_df["cln enc date"] >= one_year_ago)
]
pt_w_dbp_ov_90_df = latest_vitals_df[
    (latest_vitals_df["dia BP"] >= 90)
    & (latest_vitals_df["cln enc date"] >= one_year_ago)
]
bmi_ov_35_df = latest_vitals_df[
    (latest_vitals_df["enc BMI"] >= 35)
    & (latest_vitals_df["cln enc date"] >= one_year_ago)
]


# print(vitals_df["cln enc id"].head())
# print(enc_base_df["cln enc id"].head())
# display(bmi_ov_35_df)

# %% ---------------------------- DIAGNOSIS GROUPS, REFACTORED  ---------------------------------
enc_base_df = pd.read_csv(
    get_latest_data_file("encounter base"),
    parse_dates=["cln enc date"],
    index_col=["cln enc id"],
)
enc_dx_df = pd.read_csv(get_latest_data_file("encounter diagnoses"))

dx_groups = [
    "dm",
    "ascvd",
    "htn",
    "hlp",
    "mood",
    "learndis",
    "dementia",
    "sud",
    "sdoh",
    "noncomp",
]

# Clean dx table
dx = enc_dx_df.copy()
dx[dx_groups] = (
    dx[dx_groups]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
    .clip(0, 1)
    .astype("Int8")
)

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
pt_flags = (pt_counts >= 2).astype("Int8")


# display(pt_flags)

# %% -------------------------- MEDICATIONS -------------------------------------------

pt_meds = pd.read_csv("data/med list.csv").groupby("enterpriseid").max()
pt_meds.drop(columns=["med name"], inplace=True)

# display(pt_meds)

# %%--------------------------- KIDNEY EVALUTION IN DIABETES (KED) ---------------------


# %% ------------------------------HIGH RAF SCORE-------------------------------

raf_df = pd.read_csv(
    get_latest_data_file("raf"), index_col=["enterpriseid"], dtype={"RAF score": float}
)
9
pt_w_raf_ov_1 = raf_df[raf_df["RAF score"] >= 1]

# display( pt_w_raf_ov_1)

# %% ---------------------------------Medications------------------------------

master_df = (
    demographics_df.merge(pt_flags, on="enterpriseid", how="left")
    .merge(pt_meds, on="enterpriseid", how="left")
    .assign(
        mammogram_complete=demographics_df.index.isin(
            recent_mammo_df["enterpriseid"]
        ).astype(int),
        colonoscopy_complete=demographics_df.index.isin(colo_complete).astype(int),
        a1c_complete=demographics_df.index.isin(recent_a1c_df["enterpriseid"]).astype(
            int
        ),
        wellness_complete=demographics_df.index.isin(
            enc_w_awv_df["enterpriseid"]
        ).astype(int),
    )
)
# display(master_df)

# DEFINE TARGETS AND PREDICTORS
# Base: demographics ---
base = demographics_df.copy()

# Preventive care target flags (1 = completed, 0 = missing) ---
base["awv_complete"] = base.index.isin(enc_w_awv_df["enterpriseid"]).astype(np.int8)
base["mammogram_complete"] = base.index.isin(recent_mammo_df["enterpriseid"]).astype(
    np.int8
)
base["colorectal_complete"] = base.index.isin(colo_complete).astype(np.int8)
base["a1c_complete"] = base.index.isin(recent_a1c_df["enterpriseid"]).astype(np.int8)

# Combine into a single target: missed any preventive care ---
# (can make one binary target, or model each separately)
base["any_missed_preventive"] = 1 - (
    (
        base[
            [
                "awv_complete",
                "mammogram_complete",
                "colorectal_complete",
                "a1c_complete",
            ]
        ].sum(axis=1)
        > 0
    ).astype(np.int8)
)

# Merge with comorbidities, meds, and RAF ---
model_df = (
    base.merge(
        pt_flags, on="enterpriseid", how="left"
    )  # diagnostic groups (dm, htn, etc.)
    .merge(pt_meds, on="enterpriseid", how="left")  # med categories
    .merge(raf_df, on="enterpriseid", how="left")  # RAF score
)

# Clean and prepare predictors ---
# Fill missing numeric fields with 0, categorical with "Unknown"
num_cols = model_df.select_dtypes(include=[np.number]).columns
cat_cols = model_df.select_dtypes(exclude=[np.number]).columns

model_df[num_cols] = model_df[num_cols].fillna(0)
model_df[cat_cols] = model_df[cat_cols].fillna("Unknown")

# Define features (X) and targets (y) ---
target_cols = [
    "awv_complete",
    "mammogram_complete",
    "colorectal_complete",
    "a1c_complete",
    "any_missed_preventive",
]

predictor_cols = [
    "age",
    "patientsex",
    "RAF score",
    # diagnostic flags from pt_flags
    "dm",
    "ascvd",
    "htn",
    "hlp",
    "mood",
    "learndis",
    "dementia",
    "sud",
    "sdoh",
    "noncomp",
    # you can also add vitals, BMI, or visit counts if available
]

# Example: encode sex as 1/0 for modeling
model_df["sex_male"] = (model_df["patientsex"].str.upper().str.startswith("M")).astype(
    "Int8"
)

# Split into X, y for a sample target ---
X = model_df[predictor_cols + ["sex_male"]]
y = model_df["any_missed_preventive"]

# Quick sanity checks ---
# print("Data shape:", X.shape)
# print("Target distribution:")
# print(y.value_counts(dropna=False))
# print("\nSample predictors:\n", X.head(5))

# Optionally save the dataset for modeling
# model_df.to_csv("data/model_input.csv", index_label="enterpriseid")


#  FEATURE ENGINEERING + PREPROCESSING PIPELINE
# ---------- Helpers ----------
def ensure_col(df, col, fill):
    """Ensure df[col] exists; if missing, create and fill with a constant."""
    if col not in df.columns:
        df[col] = fill
    return df


def safe_merge(left, right, how="left", on="enterpriseid"):
    """Merge on enterpriseid whether it's an index or column."""
    l = (
        left.reset_index().rename(columns={"index": "enterpriseid"})
        if left.index.name == "enterpriseid"
        else left
    )
    r = (
        right.reset_index().rename(columns={"index": "enterpriseid"})
        if right.index.name == "enterpriseid"
        else right
    )
    return l.merge(r, on=on, how=how)


def safe_numeric(df, cols):
    """Cast to numeric where possible, leaving others intact."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------- 3A. Start from model_df built in Step 2 ----------
try:
    _ = model_df  # will raise if not defined
except NameError:
    # minimal fallback if someone runs this cell standalone; assumes Step 2 variables exist
    base = demographics_df.copy()
    base["awv_complete"] = base.index.isin(enc_w_awv_df["enterpriseid"]).astype(np.int8)
    base["mammogram_complete"] = base.index.isin(
        recent_mammo_df["enterpriseid"]
    ).astype(np.int8)
    base["colorectal_complete"] = base.index.isin(colo_complete).astype(np.int8)
    base["a1c_complete"] = base.index.isin(recent_a1c_df["enterpriseid"]).astype(
        np.int8
    )
    base["any_missed_preventive"] = 1 - (
        (
            base[
                [
                    "awv_complete",
                    "mammogram_complete",
                    "colorectal_complete",
                    "a1c_complete",
                ]
            ].sum(axis=1)
            > 0
        ).astype(np.int8)
    )
    model_df = (
        base.merge(pt_flags, on="enterpriseid", how="left")
        .merge(pt_meds, on="enterpriseid", how="left")
        .merge(raf_df, on="enterpriseid", how="left")
    )

# ---------- 3B. Clinical/engagement feature engineering ----------
fe = model_df.copy()

# Normalize key columns
fe = safe_numeric(fe, ["age", "RAF score"])

# Sex -> binary feature (keep original categorical; add model-friendly flag)
fe["patientsex"] = fe.get("patientsex", "Unknown").fillna("Unknown")
fe["sex_male"] = (
    fe["patientsex"].astype(str).str.upper().str.startswith("M").astype(np.int8)
)

# (i) Encounter recency + utilization (requires enc_base_df with 'cln enc date' and 'enterpriseid')
if "enc_base_df" in globals():
    enc_tmp = enc_base_df.copy()
    enc_tmp["cln enc date"] = pd.to_datetime(enc_tmp["cln enc date"], errors="coerce")

    # Last encounter date and days since last encounter
    last_enc = (
        enc_tmp.sort_values(["enterpriseid", "cln enc date"])
        .groupby("enterpriseid", as_index=True)["cln enc date"]
        .max()
        .to_frame("last_enc_date")
    )

    # Visits in last 12 months
    enc_12mo = enc_tmp.loc[enc_tmp["cln enc date"] >= one_year_ago]
    visit_counts_12mo = (
        enc_12mo.groupby("enterpriseid").size().rename("visit_count_12mo").to_frame()
    )

    util = last_enc.join(visit_counts_12mo, how="left").reset_index()
    util["days_since_last_enc"] = (
        pd.Timestamp.today().normalize() - util["last_enc_date"]
    ).dt.days
    util = util.fillna(
        {"visit_count_12mo": 0, "days_since_last_enc": 1e6}
    )  # large sentinel if no history

    fe = safe_merge(
        fe,
        util[
            ["enterpriseid", "last_enc_date", "visit_count_12mo", "days_since_last_enc"]
        ],
    )
else:
    fe["visit_count_12mo"] = 0
    fe["days_since_last_enc"] = 1e6

# (ii) Latest vitals → BP flags & BMI bins (requires latest_vitals_df from your Step 1 code)
if "latest_vitals_df" in globals():
    vit = latest_vitals_df.copy()
    vit = vit.sort_values(["enterpriseid", "cln enc date"]).drop_duplicates(
        "enterpriseid", keep="last"
    )

    # standardize column names present in your file
    for col in ["sys BP", "dia BP", "enc BMI"]:
        if col not in vit.columns:
            vit[col] = np.nan

    vit_feats = vit[["enterpriseid", "sys BP", "dia BP", "enc BMI"]].copy()

    # Hypertension flags (recent reading thresholds)
    vit_feats["bp_systolic_high"] = (
        (vit_feats["sys BP"] >= 140).astype("float").fillna(0.0)
    )
    vit_feats["bp_diastolic_high"] = (
        (vit_feats["dia BP"] >= 90).astype("float").fillna(0.0)
    )

    # BMI categories (Under/Normal/Over/Obese/SevereObese)
    bmi = pd.to_numeric(vit_feats["enc BMI"], errors="coerce")
    vit_feats["bmi_cat"] = (
        pd.cut(
            bmi,
            bins=[-np.inf, 18.5, 25, 30, 35, np.inf],
            labels=["Under", "Normal", "Over", "Obese", "SevereObese"],
        )
        .astype("object")
        .cat.add_categories(["Unknown"])
        .fillna("Unknown")
    )

    fe = safe_merge(fe, vit_feats, how="left")
else:
    fe["bp_systolic_high"] = 0.0
    fe["bp_diastolic_high"] = 0.0
    fe["bmi_cat"] = "Unknown"

# (iii) Comorbidity counts (from pt_flags: dm, htn, etc.)
dx_cols = [
    c
    for c in [
        "dm",
        "ascvd",
        "htn",
        "hlp",
        "mood",
        "learndis",
        "dementia",
        "sud",
        "sdoh",
        "noncomp",
    ]
    if c in fe.columns
]
for c in dx_cols:
    fe[c] = pd.to_numeric(fe[c], errors="coerce").fillna(0).clip(0, 1)
fe["comorbidity_count"] = fe[dx_cols].sum(axis=1) if dx_cols else 0

# (iv) Medication burden proxy (if pt_meds was one-hot by class)
if "pt_meds" in globals():
    med_cols = [
        c for c in fe.columns if c not in model_df.columns or c in pt_meds.columns
    ]  # conservative
    # Better heuristic: count non-null/positive indicators in pt_meds-known columns if available
    if "pt_meds" in globals():
        med_cols = [c for c in pt_meds.columns if c in fe.columns]
    fe["med_count_flags"] = (
        fe[med_cols].select_dtypes(include=[np.number]).gt(0).sum(axis=1)
        if med_cols
        else 0
    )
else:
    fe["med_count_flags"] = 0

# (v) Cross-measure engagement features (completed other measures)
completed_cols = [
    c
    for c in [
        "awv_complete",
        "mammogram_complete",
        "colorectal_complete",
        "a1c_complete",
    ]
    if c in fe.columns
]
fe["completed_measures_total"] = fe[completed_cols].sum(axis=1) if completed_cols else 0

# ---------- 3C. Minimal predictor set (extend as needed) ----------
# Categorical candidates you may have (add/trim depending on your exports)
possible_cats = ["patientsex", "bmi_cat", "insurance", "race", "ethnicity"]
for c in possible_cats:
    fe = ensure_col(fe, c, "Unknown")
    fe[c] = fe[c].astype("object").fillna("Unknown")

numeric_features = [
    # demographics
    "age",
    "RAF score",
    # utilization
    "visit_count_12mo",
    "days_since_last_enc",
    # vitals flags
    "bp_systolic_high",
    "bp_diastolic_high",
    # burden
    "comorbidity_count",
    "med_count_flags",
    # engagement
    "completed_measures_total",
]
numeric_features = [c for c in numeric_features if c in fe.columns]

categorical_features = [c for c in possible_cats if c in fe.columns]

# Keep dx flags as numeric features too (strong signals)
numeric_features += dx_cols

# Add the binary sex helper (from Step 2) if present
if "sex_male" in fe.columns:
    numeric_features += ["sex_male"]

# ---------- 3D. Choose a target and build train/test ----------
# swap to any of: "mammogram_complete", "colorectal_complete", "a1c_complete", "awv_complete"
TARGET = "any_missed_preventive"
if TARGET not in fe.columns:
    raise ValueError(f"Target column '{TARGET}' not found; ensure Step 2 created it.")

X = fe[numeric_features + categorical_features].copy()
y = fe[TARGET].astype(int)

# ---------- 3E. Preprocess: scale numeric, one-hot encode categoricals ----------
numeric_transformer = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler(with_mean=False),
        )  # with_mean=False keeps sparse compatibility
    ]
)

categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
    sparse_threshold=1.0,  # keep sparse
)

# This transformed X is now ready to feed into LogisticRegression, XGBClassifier, etc.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")
print("Target distribution (train):")
print(y_train.value_counts(normalize=True).rename("share"))

# ---------- 3F. Example: build a modeling pipeline (plug in any estimator) ----------
#   Swap in LogisticRegression or XGBClassifier in Step 4
from xgboost import XGBClassifier

clf = Pipeline(
    steps=[
        ("pre", preprocessor),
        (
            "model",
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.08,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
            ),
        ),
    ]
)

# this just shows the end-to-end object is ready:
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# y_proba = clf.predict_proba(X_test)[:,1]


# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from sklearn.metrics import roc_auc_score, f1_score, classification_report
# from xgboost import XGBClassifier

# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# model = XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=200)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred))

# import shap
# explainer = shap.Explainer(model, X_train)
# shap.summary_plot(explainer(X_test))

# import streamlit as st
# st.title("Preventive Care Risk Dashboard")
# st.metric("At-Risk Patients", f"{master_df['predicted_risk'].sum()}")

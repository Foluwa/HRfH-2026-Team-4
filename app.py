import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score


# ============================================================
# PAGE CONFIG
# ============================================================


st.set_page_config(
    page_title="Knee Recovery Simulation Dashboard",
    layout="wide"
)

HACKATHON_LOGO_URL = "https://health-research-from-home.github.io/HRfH-Hackathon-2026/images/2026_Hackathon.png"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    patients_data = pd.read_csv("synthetic_patients.csv")

    try:
        daily_truth_data = pd.read_csv("synthetic_daily_truth.csv")
    except FileNotFoundError:
        daily_truth_data = pd.read_csv("synthetic_daily_data.csv")

    return patients_data, daily_truth_data


patients_df, daily_truth_df = load_data()


# ============================================================
# NORMALISE COLUMN NAMES
# ============================================================

if "day_since_treatment" in daily_truth_df.columns and "day" not in daily_truth_df.columns:
    daily_truth_df = daily_truth_df.rename(columns={"day_since_treatment": "day"})

if "avg_hr" in daily_truth_df.columns and "heart_rate" not in daily_truth_df.columns:
    daily_truth_df = daily_truth_df.rename(columns={"avg_hr": "heart_rate"})

if "heart_rate" not in daily_truth_df.columns:
    daily_truth_df["heart_rate"] = np.nan

if "sleep_hours" not in daily_truth_df.columns:
    daily_truth_df["sleep_hours"] = np.nan

if "phenotype" not in daily_truth_df.columns and "phenotype" in patients_df.columns:
    daily_truth_df = daily_truth_df.merge(
        patients_df[["patient_id", "phenotype"]],
        on="patient_id",
        how="left"
    )

if "latent_state" not in daily_truth_df.columns:
    daily_truth_df["latent_state"] = "unknown"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_available_target_columns(data):
    possible_columns = ["steps", "heart_rate", "sleep_hours"]
    return [column for column in possible_columns if column in data.columns]


def apply_missing_values(output_data, missing_mask, target_columns):
    for column in target_columns:
        if column in output_data.columns:
            output_data.loc[missing_mask, column] = np.nan
    return output_data


def calculate_actual_missingness(data, target_column="steps"):
    if target_column not in data.columns:
        return np.nan
    return data[target_column].isna().mean() * 100


def calculate_missingness_by_column(data, target_columns):
    records = []

    for column in target_columns:
        if column in data.columns:
            records.append({
                "variable": column,
                "missingness_percent": data[column].isna().mean() * 100,
                "missing_records": int(data[column].isna().sum()),
                "total_records": int(len(data))
            })

    return pd.DataFrame(records)


def attach_patient_context(daily_data, patients_data):
    context_columns = [
        "patient_id",
        "phenotype",
        "age",
        "BMI",
        "preop_activity",
        "baseline_pain",
        "device_brand",
        "battery_life_hours",
        "last_active_days",
        "surgery_season"
    ]

    context_columns = [
        column for column in context_columns
        if column in patients_data.columns
    ]

    return daily_data.merge(
        patients_data[context_columns],
        on="patient_id",
        how="left",
        suffixes=("", "_patient")
    )


def build_group_level_summary(patients_data, daily_data):
    patient_missingness = (
        daily_data
        .assign(step_missing=daily_data["steps"].isna())
        .groupby("patient_id")
        .agg(
            mean_steps=("steps", "mean"),
            max_steps_observed=("steps", "max"),
            step_missingness_percent=("step_missing", lambda values: values.mean() * 100)
        )
        .reset_index()
    )

    summary_data = patients_data.merge(
        patient_missingness,
        on="patient_id",
        how="left"
    )

    summary = (
        summary_data
        .groupby("phenotype")
        .agg(
            patients=("patient_id", "nunique"),
            mean_age=("age", "mean"),
            mean_bmi=("BMI", "mean"),
            mean_baseline_pain=("baseline_pain", "mean"),
            mean_max_steps=("max_steps", "mean"),
            mean_observed_steps=("mean_steps", "mean"),
            mean_step_missingness_percent=("step_missingness_percent", "mean")
        )
        .reset_index()
    )

    numeric_columns = summary.select_dtypes(include=[np.number]).columns
    summary[numeric_columns] = summary[numeric_columns].round(2)

    return summary


# ============================================================
# MISSINGNESS FUNCTIONS
# ============================================================

def apply_mcar_missingness(data, missingness_percent, target_columns, random_seed=42):
    """
    MCAR: Missing Completely At Random.
    Missingness does not depend on patient features, device features,
    recovery phenotype, season, or outcome values.
    """
    output_data = data.copy()
    rng = np.random.default_rng(random_seed)

    missing_probability = missingness_percent / 100
    missing_mask = rng.random(len(output_data)) < missing_probability

    output_data = apply_missing_values(
        output_data,
        missing_mask,
        target_columns
    )

    return output_data


def apply_mar_missingness(data, patients_data, missingness_percent, target_columns, random_seed=42):
    """
    MAR: Missing At Random.
    Missingness depends on observed variables such as age, phenotype,
    device brand, battery life, season, or last active days.
    """
    output_data = data.copy()

    merge_columns = [
        "patient_id",
        "age",
        "phenotype",
        "battery_life_hours",
        "device_brand",
        "surgery_season",
        "last_active_days"
    ]

    merge_columns = [
        column for column in merge_columns
        if column in patients_data.columns
    ]

    output_data = output_data.merge(
        patients_data[merge_columns],
        on="patient_id",
        how="left",
        suffixes=("", "_patient")
    )

    if "phenotype_patient" in output_data.columns and "phenotype" not in output_data.columns:
        output_data["phenotype"] = output_data["phenotype_patient"]

    rng = np.random.default_rng(random_seed)

    base_missing_probability = missingness_percent / 100
    risk_score = np.zeros(len(output_data))

    if "phenotype" in output_data.columns:
        risk_score += np.where(output_data["phenotype"] == "slow", 0.08, 0)
        risk_score += np.where(output_data["phenotype"] == "intermittent", 0.05, 0)

    if "age" in output_data.columns:
        risk_score += np.where(output_data["age"] >= 75, 0.05, 0)

    if "battery_life_hours" in output_data.columns:
        risk_score += np.where(output_data["battery_life_hours"] < 18, 0.06, 0)

    if "surgery_season" in output_data.columns:
        risk_score += np.where(output_data["surgery_season"] == "winter", 0.04, 0)

    if "device_brand" in output_data.columns:
        risk_score += np.where(output_data["device_brand"] == "Garmin", 0.02, 0)

    if "last_active_days" in output_data.columns:
        risk_score += np.where(output_data["last_active_days"] >= 5, 0.05, 0)

    missing_probability = np.clip(
        base_missing_probability + risk_score,
        0,
        0.95
    )

    missing_mask = rng.random(len(output_data)) < missing_probability

    output_data = apply_missing_values(
        output_data,
        missing_mask,
        target_columns
    )

    columns_to_drop = [
        "age",
        "battery_life_hours",
        "device_brand",
        "surgery_season",
        "last_active_days",
        "phenotype_patient"
    ]

    output_data = output_data.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    return output_data


def apply_mnar_missingness(data, missingness_percent, target_columns, random_seed=42):
    """
    MNAR: Missing Not At Random.
    Missingness depends on the outcome or latent recovery condition.
    Here, low-step days and flare states are more likely to be missing.
    """
    output_data = data.copy()
    rng = np.random.default_rng(random_seed)

    base_missing_probability = missingness_percent / 100

    step_values = output_data["steps"].fillna(0)

    low_step_risk = np.where(step_values < 1000, 0.20, 0)
    moderate_step_risk = np.where(
        (step_values >= 1000) & (step_values < 3000),
        0.08,
        0
    )

    flare_risk = np.zeros(len(output_data))

    if "latent_state" in output_data.columns:
        flare_risk += np.where(output_data["latent_state"] == "flare", 0.15, 0)

    missing_probability = np.clip(
        base_missing_probability
        + low_step_risk
        + moderate_step_risk
        + flare_risk,
        0,
        0.95
    )

    missing_mask = rng.random(len(output_data)) < missing_probability

    output_data = apply_missing_values(
        output_data,
        missing_mask,
        target_columns
    )

    return output_data


def apply_block_missingness(data, missingness_percent, target_columns, random_seed=42):
    """
    Block/gap missingness.
    Missingness occurs in consecutive runs, simulating device non-use,
    syncing failure, battery failure, or disengagement.
    """
    output_data = data.copy()
    rng = np.random.default_rng(random_seed)

    target_missing_probability = missingness_percent / 100
    output_data["missing_block"] = False

    for patient_id in output_data["patient_id"].unique():
        patient_index = output_data[output_data["patient_id"] == patient_id].index
        number_of_patient_days = len(patient_index)

        approximate_missing_days = int(
            number_of_patient_days * target_missing_probability
        )

        days_marked_missing = 0

        while days_marked_missing < approximate_missing_days:
            block_start = rng.integers(0, number_of_patient_days)
            block_length = rng.integers(3, 21)

            block_end = min(
                block_start + block_length,
                number_of_patient_days
            )

            selected_indices = patient_index[block_start:block_end]

            output_data.loc[selected_indices, "missing_block"] = True

            days_marked_missing += len(selected_indices)

            if days_marked_missing >= number_of_patient_days:
                break

    missing_mask = output_data["missing_block"]

    output_data = apply_missing_values(
        output_data,
        missing_mask,
        target_columns
    )

    output_data = output_data.drop(columns=["missing_block"])

    return output_data


def apply_mixed_missingness(data, patients_data, missingness_percent, target_columns, random_seed=42):
    """
    Mixed real-world missingness.
    Combines MCAR, MAR, MNAR, and block/gap missingness.
    """
    mcar_percent = missingness_percent * 0.25
    mar_percent = missingness_percent * 0.25
    mnar_percent = missingness_percent * 0.25
    block_percent = missingness_percent * 0.25

    output_data = data.copy()

    output_data = apply_mcar_missingness(
        output_data,
        mcar_percent,
        target_columns,
        random_seed=random_seed + 1
    )

    output_data = apply_mar_missingness(
        output_data,
        patients_data,
        mar_percent,
        target_columns,
        random_seed=random_seed + 2
    )

    output_data = apply_mnar_missingness(
        output_data,
        mnar_percent,
        target_columns,
        random_seed=random_seed + 3
    )

    output_data = apply_block_missingness(
        output_data,
        block_percent,
        target_columns,
        random_seed=random_seed + 4
    )

    return output_data


def apply_missingness(data, patients_data, missingness_type, missingness_percent, target_columns, random_seed=42):
    if missingness_type == "MCAR":
        return apply_mcar_missingness(
            data,
            missingness_percent,
            target_columns,
            random_seed
        )

    if missingness_type == "MAR":
        return apply_mar_missingness(
            data,
            patients_data,
            missingness_percent,
            target_columns,
            random_seed
        )

    if missingness_type == "MNAR":
        return apply_mnar_missingness(
            data,
            missingness_percent,
            target_columns,
            random_seed
        )

    if missingness_type == "Block / gap missingness":
        return apply_block_missingness(
            data,
            missingness_percent,
            target_columns,
            random_seed
        )

    if missingness_type == "Mixed real-world":
        return apply_mixed_missingness(
            data,
            patients_data,
            missingness_percent,
            target_columns,
            random_seed
        )

    return data.copy()


# ============================================================
# CLUSTERING FUNCTIONS
# ============================================================

def create_patient_level_features(daily_data, patients_data):
    analysis_data = daily_data.copy()

    analysis_data["steps_clean"] = analysis_data["steps"]

    patient_features = (
        analysis_data
        .groupby("patient_id")
        .agg(
            mean_steps=("steps_clean", "mean"),
            median_steps=("steps_clean", "median"),
            max_steps_observed=("steps_clean", "max"),
            step_variability=("steps_clean", "std"),
            missingness_rate=("steps_clean", lambda values: values.isna().mean()),
            mean_heart_rate=("heart_rate", "mean"),
            mean_sleep_hours=("sleep_hours", "mean")
        )
        .reset_index()
    )

    key_days = [30, 90, 180, 364]

    for key_day in key_days:
        day_data = analysis_data[analysis_data["day"] == key_day][
            ["patient_id", "steps_clean"]
        ].rename(columns={"steps_clean": f"steps_day_{key_day}"})

        patient_features = patient_features.merge(
            day_data,
            on="patient_id",
            how="left"
        )

    early_period = analysis_data[
        analysis_data["day"].between(1, 90)
    ]

    late_period = analysis_data[
        analysis_data["day"].between(91, 364)
    ]

    early_mean = (
        early_period
        .groupby("patient_id")["steps_clean"]
        .mean()
        .reset_index(name="early_mean_steps")
    )

    late_mean = (
        late_period
        .groupby("patient_id")["steps_clean"]
        .mean()
        .reset_index(name="late_mean_steps")
    )

    patient_features = patient_features.merge(
        early_mean,
        on="patient_id",
        how="left"
    )

    patient_features = patient_features.merge(
        late_mean,
        on="patient_id",
        how="left"
    )

    patient_features["recovery_change"] = (
        patient_features["late_mean_steps"]
        - patient_features["early_mean_steps"]
    )

    if "phenotype" in patients_data.columns:
        patient_features = patient_features.merge(
            patients_data[["patient_id", "phenotype"]],
            on="patient_id",
            how="left"
        )

    return patient_features


def run_recovery_clustering(daily_data, patients_data, number_of_clusters):
    patient_features = create_patient_level_features(
        daily_data,
        patients_data
    )

    numeric_columns = [
        "mean_steps",
        "median_steps",
        "max_steps_observed",
        "step_variability",
        "missingness_rate",
        "mean_heart_rate",
        "mean_sleep_hours",
        "steps_day_30",
        "steps_day_90",
        "steps_day_180",
        "steps_day_364",
        "early_mean_steps",
        "late_mean_steps",
        "recovery_change"
    ]

    numeric_columns = [
        column for column in numeric_columns
        if column in patient_features.columns
    ]

    clustering_data = patient_features[numeric_columns].copy()
    clustering_data = clustering_data.replace([np.inf, -np.inf], np.nan)

    # Some columns can become entirely missing under high missingness settings.
    # SimpleImputer with median handles ordinary missingness; if a whole column
    # is missing, we fall back to 0 so KMeans always receives finite values.
    clustering_data = clustering_data.dropna(axis=1, how="all")

    if clustering_data.empty:
        patient_features["cluster"] = "0"
        patient_features["pca_1"] = 0.0
        patient_features["pca_2"] = 0.0
        return patient_features

    imputer = SimpleImputer(strategy="median")
    imputed_features = imputer.fit_transform(clustering_data)

    imputed_features = np.nan_to_num(
        imputed_features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(imputed_features)

    scaled_features = np.nan_to_num(
        scaled_features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    kmeans = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10
    )

    patient_features["cluster"] = kmeans.fit_predict(scaled_features).astype(str)

    pca = PCA(n_components=2, random_state=42)
    pca_coordinates = pca.fit_transform(scaled_features)

    patient_features["pca_1"] = pca_coordinates[:, 0]
    patient_features["pca_2"] = pca_coordinates[:, 1]

    return patient_features


# ============================================================
# MONTE CARLO FUNCTIONS
# ============================================================

def run_monte_carlo_evaluation(
    truth_data,
    patients_data,
    missingness_type,
    missingness_percent,
    target_columns,
    number_of_simulations,
    target_day
):
    monte_carlo_records = []

    true_mean_at_target_day = (
        truth_data
        .loc[truth_data["day"] == target_day, "steps"]
        .mean()
    )

    for simulation_id in range(1, number_of_simulations + 1):
        simulated_data = apply_missingness(
            data=truth_data,
            patients_data=patients_data,
            missingness_type=missingness_type,
            missingness_percent=missingness_percent,
            target_columns=target_columns,
            random_seed=1000 + simulation_id
        )

        observed_mean_at_target_day = (
            simulated_data
            .loc[simulated_data["day"] == target_day, "steps"]
            .mean()
        )

        missingness_rate = simulated_data["steps"].isna().mean()

        bias = observed_mean_at_target_day - true_mean_at_target_day

        monte_carlo_records.append({
            "simulation_id": simulation_id,
            "target_day": target_day,
            "true_mean_steps": true_mean_at_target_day,
            "observed_mean_steps": observed_mean_at_target_day,
            "bias": bias,
            "missingness_rate": missingness_rate
        })

    monte_carlo_results = pd.DataFrame(monte_carlo_records)

    summary = {
        "mean_observed_steps": monte_carlo_results["observed_mean_steps"].mean(),
        "mcse_observed_steps": monte_carlo_results["observed_mean_steps"].std(ddof=1) / np.sqrt(number_of_simulations),
        "mean_bias": monte_carlo_results["bias"].mean(),
        "mcse_bias": monte_carlo_results["bias"].std(ddof=1) / np.sqrt(number_of_simulations),
        "rmse": np.sqrt(np.mean(monte_carlo_results["bias"] ** 2)),
        "mean_missingness_rate": monte_carlo_results["missingness_rate"].mean()
    }

    return monte_carlo_results, summary


def run_monte_carlo_trajectory_evaluation(
    truth_data,
    patients_data,
    missingness_type,
    missingness_percent,
    target_columns,
    number_of_simulations
):
    trajectory_records = []

    true_mean_by_day = (
        truth_data
        .groupby("day")["steps"]
        .mean()
        .reset_index(name="true_mean_steps")
    )

    for simulation_id in range(1, number_of_simulations + 1):
        simulated_data = apply_missingness(
            data=truth_data,
            patients_data=patients_data,
            missingness_type=missingness_type,
            missingness_percent=missingness_percent,
            target_columns=target_columns,
            random_seed=2000 + simulation_id
        )

        observed_mean_by_day = (
            simulated_data
            .groupby("day")["steps"]
            .mean()
            .reset_index(name="observed_mean_steps")
        )

        observed_mean_by_day["simulation_id"] = simulation_id

        trajectory_records.append(observed_mean_by_day)

    trajectory_results = pd.concat(
        trajectory_records,
        ignore_index=True
    )

    trajectory_summary = (
        trajectory_results
        .groupby("day")["observed_mean_steps"]
        .agg(
            mean_observed_steps="mean",
            simulation_sd="std",
            number_of_simulations="count"
        )
        .reset_index()
    )

    trajectory_summary["mcse"] = (
        trajectory_summary["simulation_sd"]
        / np.sqrt(trajectory_summary["number_of_simulations"])
    )

    trajectory_summary = trajectory_summary.merge(
        true_mean_by_day,
        on="day",
        how="left"
    )

    trajectory_summary["bias"] = (
        trajectory_summary["mean_observed_steps"]
        - trajectory_summary["true_mean_steps"]
    )

    trajectory_summary["lower_mcse_band"] = (
        trajectory_summary["mean_observed_steps"]
        - 1.96 * trajectory_summary["mcse"]
    )

    trajectory_summary["upper_mcse_band"] = (
        trajectory_summary["mean_observed_steps"]
        + 1.96 * trajectory_summary["mcse"]
    )

    return trajectory_summary


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.title("Dashboard Controls")

missingness_type = st.sidebar.selectbox(
    "Missingness mechanism",
    [
        "MCAR",
        "MAR",
        "MNAR",
        "Block / gap missingness",
        "Mixed real-world"
    ]
)

missingness_percent = st.sidebar.slider(
    "Target missingness percentage",
    min_value=0,
    max_value=80,
    value=20,
    step=5
)

available_target_columns = get_available_target_columns(daily_truth_df)

target_columns = st.sidebar.multiselect(
    "Apply missingness to",
    available_target_columns,
    default=[column for column in ["steps", "heart_rate"] if column in available_target_columns]
)

random_seed = st.sidebar.number_input(
    "Random seed",
    min_value=1,
    max_value=9999,
    value=42,
    step=1
)

selected_phenotypes = st.sidebar.multiselect(
    "Select phenotype",
    sorted(patients_df["phenotype"].dropna().unique()),
    default=sorted(patients_df["phenotype"].dropna().unique())
)

available_genders = sorted(patients_df["gender"].dropna().unique()) if "gender" in patients_df.columns else []

selected_genders = st.sidebar.multiselect(
    "Select gender",
    available_genders,
    default=available_genders
)

if "age" in patients_df.columns:
    minimum_age = int(patients_df["age"].min())
    maximum_age = int(patients_df["age"].max())

    selected_age_range = st.sidebar.slider(
        "Select age range",
        min_value=minimum_age,
        max_value=maximum_age,
        value=(minimum_age, maximum_age),
        step=1
    )
else:
    selected_age_range = None

filtered_patients = patients_df[
    patients_df["phenotype"].isin(selected_phenotypes)
]

if "gender" in filtered_patients.columns and selected_genders:
    filtered_patients = filtered_patients[
        filtered_patients["gender"].isin(selected_genders)
    ]

if "age" in filtered_patients.columns and selected_age_range is not None:
    filtered_patients = filtered_patients[
        filtered_patients["age"].between(
            selected_age_range[0],
            selected_age_range[1]
        )
    ]

if filtered_patients.empty:
    st.warning("No patients match the selected filters. Please adjust phenotype, gender, or age range.")
    st.stop()

selected_patient_id = st.sidebar.selectbox(
    "Select patient",
    sorted(filtered_patients["patient_id"].unique())
)

number_of_clusters = st.sidebar.slider(
    "Number of recovery clusters",
    min_value=2,
    max_value=6,
    value=3,
    step=1
)


# ============================================================
# APPLY FILTERS AND MISSINGNESS
# ============================================================

filtered_truth_df = daily_truth_df[
    daily_truth_df["patient_id"].isin(filtered_patients["patient_id"])
].copy()

simulated_missing_df = apply_missingness(
    data=filtered_truth_df,
    patients_data=patients_df,
    missingness_type=missingness_type,
    missingness_percent=missingness_percent,
    target_columns=target_columns,
    random_seed=random_seed
)

patient_truth_df = filtered_truth_df[
    filtered_truth_df["patient_id"] == selected_patient_id
].copy()

patient_observed_df = simulated_missing_df[
    simulated_missing_df["patient_id"] == selected_patient_id
].copy()

patient_profile = patients_df[
    patients_df["patient_id"] == selected_patient_id
].iloc[0]


# ============================================================
# MAIN DASHBOARD
# ============================================================

logo_col, title_col = st.columns([1.2, 4.8])

with logo_col:
    st.image(
        HACKATHON_LOGO_URL,
        width=280
    )

with title_col:
    st.title("Synthetic Knee Replacement Recovery Dashboard")

st.markdown(
    """
This dashboard explores simulated post-operative recovery trajectories using wearable-style data.
It allows users to apply different missingness mechanisms, change the degree of missingness,
compare observed trajectories with the underlying simulated truth, and inspect inferred recovery clusters.
"""
)


# ============================================================
# METRICS
# ============================================================

actual_missingness = calculate_actual_missingness(
    simulated_missing_df,
    target_column="steps"
)

missingness_by_column_df = calculate_missingness_by_column(
    simulated_missing_df,
    target_columns
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Patients", filtered_patients["patient_id"].nunique())
col2.metric("Daily records", f"{len(simulated_missing_df):,}")
col3.metric("Target missingness", f"{missingness_percent}%")
col4.metric("Actual step missingness", f"{actual_missingness:.1f}%")

overview_tab, assumptions_tab, patient_tab, phenotype_tab, missingness_tab, clustering_tab, monte_carlo_tab, download_tab = st.tabs(
    [
        "Overview",
        "Assumptions",
        "Patient Explorer",
        "Phenotype Recovery",
        "Missingness",
        "Clustering",
        "Monte Carlo",
        "Downloads"
    ]
)

# ============================================================
# OVERVIEW TAB
# ============================================================

with overview_tab:
    st.subheader("Dashboard Overview")

    st.markdown(
        """
This dashboard explores simulated post-operative recovery trajectories using wearable-style data.
It allows users to apply different missingness mechanisms, change the degree of missingness,
compare observed trajectories with the underlying simulated truth, and inspect inferred recovery clusters.
"""
    )

    with st.expander("Missingness mechanism definitions", expanded=True):
        st.markdown(
            """
**MCAR — Missing Completely At Random**  
Data are removed randomly. Missingness does not depend on patient characteristics, device use, recovery phenotype, season, or step count.

**MAR — Missing At Random**  
Missingness depends on observed information such as age, recovery phenotype, device brand, battery life, last active days, or surgery season.

**MNAR — Missing Not At Random**  
Missingness depends on the outcome or recovery condition itself. In this dashboard, low-step days and flare states are more likely to be missing.

**Block / gap missingness**  
Missingness occurs in consecutive periods, representing real-world gaps such as device non-use, battery failure, syncing failure, or temporary disengagement.

**Mixed real-world**  
A combined scenario using MCAR, MAR, MNAR, and block/gap missingness.
"""
        )


    st.subheader("Current Scenario Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Missingness mechanism", missingness_type)
    summary_col2.metric("Target missingness", f"{missingness_percent}%")
    summary_col3.metric("Actual step missingness", f"{actual_missingness:.1f}%")

    st.subheader("Active Patient Filters")

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    filter_col1.write(f"**Phenotypes:** {', '.join(selected_phenotypes) if selected_phenotypes else 'None'}")
    filter_col2.write(f"**Genders:** {', '.join(selected_genders) if selected_genders else 'None'}")

    if selected_age_range is not None:
        filter_col3.write(f"**Age range:** {selected_age_range[0]}–{selected_age_range[1]}")
    else:
        filter_col3.write("**Age range:** Not available")

    st.markdown("**Selected target columns for missingness:**")
    st.write(", ".join(target_columns) if target_columns else "No target columns selected")

    st.subheader("Actual Missingness by Target Column")

    if missingness_by_column_df.empty:
        st.info("No target columns selected for missingness.")
    else:
        st.dataframe(
            missingness_by_column_df,
            use_container_width=True
        )

    st.subheader("Group-Level Summary")

    group_summary_df = build_group_level_summary(
        filtered_patients,
        simulated_missing_df
    )

    st.dataframe(
        group_summary_df,
        use_container_width=True
    )
# ============================================================
# PATIENT EXPLORER TAB
# ============================================================

# ============================================================
# ASSUMPTIONS TAB
# ============================================================

with assumptions_tab:
    st.subheader("Simulator Assumptions")

    st.markdown(
        """
The simulator is designed to produce plausible post-operative wearable-style recovery data rather than a single definitive clinical model.

**Recovery phenotypes**

- **Fast recovery:** patients recover earlier and reach a higher activity level.
- **Intermittent recovery:** patients improve overall but experience variable recovery, setbacks, or flare periods.
- **Slow recovery:** patients recover more gradually and tend to plateau at lower activity levels.

**Patient heterogeneity**

Recovery patterns are influenced by patient-level characteristics such as age, BMI, pre-operative activity, baseline pain, comorbidities, device behaviour, and surgery season.

**Day-to-day variation**

Daily steps are intentionally variable to reflect real-world wearable data. The simulation includes latent recovery states such as stable, improving, flare, and plateau.

**Missingness assumptions**

Missingness can be random, related to observed patient/device characteristics, related to low-activity or flare days, or appear as consecutive blocks of missing data.
"""
    )

    st.subheader("Known Limitations")

    st.markdown(
        """
- The dataset is synthetic and should not be interpreted as real patient data.
- The recovery phenotypes are simplified representations of complex clinical recovery.
- The missingness mechanisms are intentionally transparent and configurable, but they do not capture every possible real-world cause of missing wearable data.
- Clustering results depend on the selected missingness mechanism, missingness percentage, available variables, and number of clusters.
- Monte Carlo results describe uncertainty under the simulator assumptions, not uncertainty from real-world clinical sampling.
"""
    )


# ============================================================
# PATIENT EXPLORER TAB
# ============================================================

with patient_tab:
    st.subheader("Selected Patient Profile")

    profile_columns = [
        "patient_id",
        "age",
        "gender",
        "height_cm",
        "weight_kg",
        "BMI",
        "preop_activity",
        "baseline_pain",
        "cardiovascular",
        "diabetes",
        "hypertension",
        "osteoporosis",
        "musculoskeletal_disease",
        "device_brand",
        "battery_life_hours",
        "last_active_days",
        "phenotype",
        "max_steps",
        "surgery_date",
        "surgery_season"
    ]

    profile_columns = [
        column for column in profile_columns
        if column in patients_df.columns
    ]

    st.dataframe(
        patients_df.loc[
            patients_df["patient_id"] == selected_patient_id,
            profile_columns
        ],
        use_container_width=True
    )

    st.subheader("Patient Step Trajectory: Ground Truth vs Observed")

    patient_steps_plot_df = patient_truth_df[
        ["patient_id", "day", "steps"]
    ].rename(columns={"steps": "value"})

    patient_steps_plot_df["series"] = "Ground truth"

    observed_steps_plot_df = patient_observed_df[
        ["patient_id", "day", "steps"]
    ].rename(columns={"steps": "value"})

    observed_steps_plot_df["series"] = "Observed after missingness"

    steps_plot_df = pd.concat(
        [patient_steps_plot_df, observed_steps_plot_df],
        ignore_index=True
    )

    fig = px.line(
        steps_plot_df,
        x="day",
        y="value",
        color="series",
        title=f"Patient {selected_patient_id} Daily Steps",
        labels={
            "day": "Day post-operation",
            "value": "Daily steps",
            "series": "Series"
        },
        hover_data=["patient_id", "day", "value", "series"]
    )

    fig.update_traces(
        mode="lines+markers",
        marker=dict(size=4),
        connectgaps=False
    )

    fig.update_layout(
        height=520,
        xaxis=dict(
            range=[1, 365],
            rangeslider=dict(visible=True)
        ),
        yaxis=dict(range=[0, None]),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Patient Heart Rate Trajectory")

    fig = px.line(
        patient_observed_df,
        x="day",
        y="heart_rate",
        title=f"Patient {selected_patient_id} Heart Rate After Missingness",
        labels={
            "day": "Day post-operation",
            "heart_rate": "Heart rate"
        },
        hover_data=[
            "patient_id",
            "day",
            "heart_rate",
            "steps",
            "phenotype",
            "latent_state"
        ]
    )

    fig.update_traces(
        mode="lines+markers",
        marker=dict(size=4),
        connectgaps=False
    )

    fig.update_layout(
        height=500,
        xaxis=dict(
            range=[1, 365],
            rangeslider=dict(visible=True)
        ),
        yaxis=dict(range=[40, 130]),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PHENOTYPE RECOVERY TAB
# ============================================================

with phenotype_tab:
    st.subheader("Average Recovery by Phenotype")

    average_truth_by_phenotype = (
        filtered_truth_df
        .groupby(["phenotype", "day"])["steps"]
        .mean()
        .reset_index()
    )

    average_truth_by_phenotype["series"] = "Ground truth"

    average_observed_by_phenotype = (
        simulated_missing_df
        .groupby(["phenotype", "day"])["steps"]
        .mean()
        .reset_index()
    )

    average_observed_by_phenotype["series"] = "Observed after missingness"

    average_recovery_plot_df = pd.concat(
        [average_truth_by_phenotype, average_observed_by_phenotype],
        ignore_index=True
    )

    fig = px.line(
        average_recovery_plot_df,
        x="day",
        y="steps",
        color="phenotype",
        line_dash="series",
        title="Average Recovery by Phenotype: Truth vs Observed",
        labels={
            "day": "Day post-operation",
            "steps": "Average daily steps",
            "phenotype": "Phenotype",
            "series": "Series"
        },
        hover_data=["phenotype", "series", "day", "steps"]
    )

    fig.update_layout(
        height=560,
        xaxis=dict(
            range=[1, 365],
            rangeslider=dict(visible=True)
        ),
        yaxis=dict(range=[0, None]),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# MISSINGNESS TAB
# ============================================================

with missingness_tab:
    st.subheader("Missingness Rate Over Time")

    missingness_by_day = (
        simulated_missing_df
        .assign(is_missing=simulated_missing_df["steps"].isna())
        .groupby("day")["is_missing"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        missingness_by_day,
        x="day",
        y="is_missing",
        title="Step Missingness Rate Over Time",
        labels={
            "day": "Day post-operation",
            "is_missing": "Missingness rate"
        },
        hover_data=["day", "is_missing"]
    )

    fig.update_layout(
        height=460,
        xaxis=dict(
            range=[1, 365],
            rangeslider=dict(visible=True)
        ),
        yaxis=dict(range=[0, 1]),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Missingness by Surgery Season")

    missingness_context_df = attach_patient_context(
        simulated_missing_df,
        patients_df
    )

    if "surgery_season" in missingness_context_df.columns:
        missingness_by_season = (
            missingness_context_df
            .assign(is_missing=missingness_context_df["steps"].isna())
            .groupby("surgery_season")["is_missing"]
            .mean()
            .reset_index()
        )

        missingness_by_season["missingness_percent"] = missingness_by_season["is_missing"] * 100

        fig = px.bar(
            missingness_by_season,
            x="surgery_season",
            y="missingness_percent",
            title="Step Missingness by Surgery Season",
            labels={
                "surgery_season": "Surgery season",
                "missingness_percent": "Missingness (%)"
            },
            hover_data=["surgery_season", "missingness_percent"]
        )

        fig.update_layout(height=420, yaxis=dict(range=[0, 100]))

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Surgery season is not available in the loaded patient-level data.")

    st.subheader("Missingness by Latent Recovery State")

    if "latent_state" in simulated_missing_df.columns:
        missingness_by_state = (
            simulated_missing_df
            .assign(is_missing=simulated_missing_df["steps"].isna())
            .groupby("latent_state")["is_missing"]
            .mean()
            .reset_index()
        )

        missingness_by_state["missingness_percent"] = missingness_by_state["is_missing"] * 100

        fig = px.bar(
            missingness_by_state,
            x="latent_state",
            y="missingness_percent",
            title="Step Missingness by Latent Recovery State",
            labels={
                "latent_state": "Latent recovery state",
                "missingness_percent": "Missingness (%)"
            },
            hover_data=["latent_state", "missingness_percent"]
        )

        fig.update_layout(height=420, yaxis=dict(range=[0, 100]))

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Latent recovery state is not available in the loaded daily data.")

    st.subheader("Missingness Heatmap")

    heatmap_df = simulated_missing_df.copy()
    heatmap_df["is_missing"] = heatmap_df["steps"].isna().astype(int)

    missingness_matrix = heatmap_df.pivot(
        index="patient_id",
        columns="day",
        values="is_missing"
    )

    fig = px.imshow(
        missingness_matrix,
        aspect="auto",
        title="Step Missingness Heatmap",
        labels={
            "x": "Day post-operation",
            "y": "Patient ID",
            "color": "Missing"
        }
    )

    fig.update_layout(height=650)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Missingness by Phenotype")

    missingness_by_phenotype = (
        simulated_missing_df
        .assign(is_missing=simulated_missing_df["steps"].isna())
        .groupby("phenotype")["is_missing"]
        .mean()
        .reset_index()
    )

    missingness_by_phenotype["missingness_percent"] = missingness_by_phenotype["is_missing"] * 100

    fig = px.bar(
        missingness_by_phenotype,
        x="phenotype",
        y="missingness_percent",
        title="Step Missingness by Phenotype",
        labels={
            "phenotype": "Phenotype",
            "missingness_percent": "Missingness (%)"
        },
        hover_data=["phenotype", "missingness_percent"]
    )

    fig.update_layout(height=420, yaxis=dict(range=[0, 100]))

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# CLUSTERING TAB
# ============================================================

with clustering_tab:
    st.subheader("Recovery Trajectory Clustering")

    st.markdown(
        """
The clustering step does not use the phenotype label directly. It derives patient-level trajectory features from the observed data after missingness has been applied, then groups patients into recovery clusters. The inferred clusters are compared with the simulated phenotypes to show whether recovery patterns remain detectable under the selected missingness scenario.
"""
    )

    clustered_patients_df = run_recovery_clustering(
        simulated_missing_df,
        patients_df,
        number_of_clusters
    )

    fig = px.scatter(
        clustered_patients_df,
        x="pca_1",
        y="pca_2",
        color="cluster",
        symbol="phenotype" if "phenotype" in clustered_patients_df.columns else None,
        title="Patient Recovery Clusters Based on Observed Trajectory Features",
        labels={
            "pca_1": "PCA component 1",
            "pca_2": "PCA component 2",
            "cluster": "Inferred cluster",
            "phenotype": "True phenotype"
        },
        hover_data=[
            "patient_id",
            "phenotype",
            "cluster",
            "mean_steps",
            "max_steps_observed",
            "missingness_rate",
            "recovery_change"
        ]
    )

    fig.update_layout(height=560)

    st.plotly_chart(fig, use_container_width=True)

    cluster_composition = pd.crosstab(
        clustered_patients_df["cluster"],
        clustered_patients_df["phenotype"]
    )

    st.write("Cluster composition compared with simulated phenotype")
    st.dataframe(cluster_composition, use_container_width=True)

    cluster_composition_percent = cluster_composition.div(
        cluster_composition.sum(axis=1),
        axis=0
    ) * 100

    cluster_composition_percent = cluster_composition_percent.round(1)

    st.write("Cluster composition as percentages")
    st.dataframe(cluster_composition_percent, use_container_width=True)

    dominant_cluster_profile = cluster_composition_percent.idxmax(axis=1).reset_index()
    dominant_cluster_profile.columns = ["cluster", "dominant_phenotype"]

    st.write("Dominant phenotype by inferred cluster")
    st.dataframe(dominant_cluster_profile, use_container_width=True)

    if "phenotype" in clustered_patients_df.columns:
        phenotype_codes = clustered_patients_df["phenotype"].astype("category").cat.codes
        cluster_codes = clustered_patients_df["cluster"].astype("category").cat.codes

        adjusted_rand_index = adjusted_rand_score(
            phenotype_codes,
            cluster_codes
        )

        st.metric(
            "Adjusted Rand Index: cluster agreement with phenotype",
            f"{adjusted_rand_index:.3f}"
        )


# ============================================================
# MONTE CARLO TAB
# ============================================================

with monte_carlo_tab:
    st.subheader("Monte Carlo Evaluation")

    number_of_simulations = st.slider(
        "Number of Monte Carlo simulations",
        min_value=10,
        max_value=300,
        value=50,
        step=10
    )

    target_day = st.selectbox(
        "Target recovery day",
        [30, 90, 180, 364],
        index=1
    )

    run_monte_carlo = st.button("Run Monte Carlo evaluation")

    if run_monte_carlo:
        with st.spinner("Running Monte Carlo simulations..."):
            monte_carlo_results, monte_carlo_summary = run_monte_carlo_evaluation(
                truth_data=filtered_truth_df,
                patients_data=patients_df,
                missingness_type=missingness_type,
                missingness_percent=missingness_percent,
                target_columns=target_columns,
                number_of_simulations=number_of_simulations,
                target_day=target_day
            )

        mc_col1, mc_col2, mc_col3, mc_col4 = st.columns(4)

        mc_col1.metric(
            "Mean observed steps",
            f"{monte_carlo_summary['mean_observed_steps']:.1f}"
        )

        mc_col2.metric(
            "MCSE observed steps",
            f"{monte_carlo_summary['mcse_observed_steps']:.2f}"
        )

        mc_col3.metric(
            "Mean bias",
            f"{monte_carlo_summary['mean_bias']:.1f}"
        )

        mc_col4.metric(
            "MCSE bias",
            f"{monte_carlo_summary['mcse_bias']:.2f}"
        )

        mc_col5, mc_col6 = st.columns(2)

        mc_col5.metric(
            "RMSE",
            f"{monte_carlo_summary['rmse']:.1f}"
        )

        mc_col6.metric(
            "Mean missingness rate",
            f"{monte_carlo_summary['mean_missingness_rate'] * 100:.1f}%"
        )

        fig = px.histogram(
            monte_carlo_results,
            x="observed_mean_steps",
            nbins=30,
            title=f"Monte Carlo Distribution of Observed Mean Steps at Day {target_day}",
            labels={
                "observed_mean_steps": "Observed mean steps"
            }
        )

        fig.update_layout(height=450)

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            monte_carlo_results,
            use_container_width=True
        )

        st.subheader("Monte Carlo Trajectory-Level MCSE")

        trajectory_summary = run_monte_carlo_trajectory_evaluation(
            truth_data=filtered_truth_df,
            patients_data=patients_df,
            missingness_type=missingness_type,
            missingness_percent=missingness_percent,
            target_columns=target_columns,
            number_of_simulations=number_of_simulations
        )

        fig = px.line(
            trajectory_summary,
            x="day",
            y=["true_mean_steps", "mean_observed_steps"],
            title="Monte Carlo Mean Recovery Trajectory: Truth vs Observed",
            labels={
                "day": "Day post-operation",
                "value": "Mean daily steps",
                "variable": "Series"
            }
        )

        fig.update_layout(
            height=520,
            xaxis=dict(
                range=[1, 365],
                rangeslider=dict(visible=True)
            ),
            yaxis=dict(range=[0, None]),
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

        fig = px.line(
            trajectory_summary,
            x="day",
            y="mcse",
            title="Monte Carlo Standard Error by Day",
            labels={
                "day": "Day post-operation",
                "mcse": "MCSE"
            }
        )

        fig.update_layout(
            height=420,
            xaxis=dict(
                range=[1, 365],
                rangeslider=dict(visible=True)
            ),
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            trajectory_summary,
            use_container_width=True
        )
    else:
        st.info("Choose the number of simulations and target day, then click the button to run the Monte Carlo evaluation.")


# ============================================================
# DOWNLOADS TAB
# ============================================================

with download_tab:
    st.subheader("Download Data")

    st.markdown(
        """
Download the patient-level data, the full ground-truth daily data, or the currently active dataset after applying the selected missingness settings.
"""
    )

    download_scope = st.radio(
        "Download scope",
        [
            "All patients",
            "Filtered phenotype patients",
            "Selected patient only"
        ],
        index=1,
        horizontal=True
    )

    if download_scope == "All patients":
        patient_download_df = patients_df.copy()
        truth_download_df = daily_truth_df.copy()
        missing_download_source_df = apply_missingness(
            data=daily_truth_df,
            patients_data=patients_df,
            missingness_type=missingness_type,
            missingness_percent=missingness_percent,
            target_columns=target_columns,
            random_seed=random_seed
        )
    elif download_scope == "Selected patient only":
        patient_download_df = patients_df[
            patients_df["patient_id"] == selected_patient_id
        ].copy()
        truth_download_df = filtered_truth_df[
            filtered_truth_df["patient_id"] == selected_patient_id
        ].copy()
        missing_download_source_df = simulated_missing_df[
            simulated_missing_df["patient_id"] == selected_patient_id
        ].copy()
    else:
        patient_download_df = filtered_patients.copy()
        truth_download_df = filtered_truth_df.copy()
        missing_download_source_df = simulated_missing_df.copy()

    download_format = st.radio(
        "Missing value export format for active missingness data",
        ["Keep missing values as blank/NaN", "Convert missing values to 0"],
        horizontal=True
    )

    patient_csv = patient_download_df.to_csv(index=False).encode("utf-8")

    if download_scope == "Selected patient only":
        patient_file_name = f"synthetic_patients_patient_{selected_patient_id}.csv"
    elif download_scope == "Filtered phenotype patients":
        patient_file_name = "synthetic_patients_filtered.csv"
    else:
        patient_file_name = "synthetic_patients_all.csv"

    st.download_button(
        label="Download patient-level data",
        data=patient_csv,
        file_name=patient_file_name,
        mime="text/csv"
    )

    truth_csv = truth_download_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download full ground-truth daily data",
        data=truth_csv,
        file_name="synthetic_daily_truth.csv",
        mime="text/csv"
    )

    active_missing_df = missing_download_source_df.copy()

    if download_format == "Convert missing values to 0":
        for column in available_target_columns:
            if column in active_missing_df.columns:
                active_missing_df[column] = active_missing_df[column].fillna(0)

    active_missing_csv = active_missing_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download currently active missingness data",
        data=active_missing_csv,
        file_name=f"simulated_data_{missingness_type}_{missingness_percent}_missing.csv",
        mime="text/csv"
    )

    st.subheader("Preview of Active Missingness Data")

    st.dataframe(
        active_missing_df.head(100),
        use_container_width=True
    )
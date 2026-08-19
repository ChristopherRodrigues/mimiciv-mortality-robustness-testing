import os
import sys
import copy
import glob
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
import importlib.util

from scipy.stats import spearmanr

from mimic4benchmark.readers import InHospitalMortalityReader
from mimic4models import common_utils


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:/Users/chris/Thesis/mimic4-benchmarks"

DATA_DIR = os.path.join(
    BASE_DIR,
    "data/in-hospital-mortality"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "sicu_shap"
)

sys.path.append(
    os.path.join(
        BASE_DIR,
        "mimic4models"
    )
)


# ============================================================
# SICU UNIT
# ============================================================

SICU_UNIT = "Surgical Intensive Care Unit (SICU)"


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 42

np.random.seed(
    RANDOM_SEED
)

tf.random.set_seed(
    RANDOM_SEED
)


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(x):

    try:

        v = float(x)

        if np.isfinite(v):
            return v

    except Exception:
        pass

    return np.nan


# ============================================================
# ICU MAP
# ============================================================

def build_icu_map(root_path):

    icu_map = {}

    for split in ["train", "test"]:

        split_path = os.path.join(
            root_path,
            split
        )

        if not os.path.exists(split_path):
            continue

        for subject_id in os.listdir(split_path):

            subject_path = os.path.join(
                split_path,
                subject_id
            )

            if not os.path.isdir(subject_path):
                continue

            stays_file = os.path.join(
                subject_path,
                "stays.csv"
            )

            if not os.path.exists(stays_file):
                continue

            stays_df = pd.read_csv(
                stays_file
            )

            if "intime" in stays_df.columns:

                stays_df = stays_df.sort_values(
                    by="intime"
                )

            if "LAST_CAREUNIT" not in stays_df.columns:
                continue

            icu_units = stays_df[
                "LAST_CAREUNIT"
            ].tolist()

            for i, icu in enumerate(icu_units):

                episode_idx = i + 1

                key = (
                    f"{subject_id}"
                    f"_episode{episode_idx}"
                    f"_timeseries.csv"
                )

                icu_map[key] = icu

    return icu_map


# ============================================================
# SICU PERTURBATION PARAMETERS
# ============================================================

SICU_PARAMS = {

    "Minimal": {

        "temp_shift": 0.3,
        "hr_pct": 0.05,
        "rr_pct": 0.05,
        "wbc_pct": 0.15,
        "lactate_pct": 0.0,
        "hgb_pct": -0.05,
        "hct_pct": -0.05,
        "fio2_pct": 0.0,
        "creat_pct": 0.0,
    },

    "Moderate": {

        "temp_shift": 0.8,
        "hr_pct": 0.10,
        "rr_pct": 0.10,
        "wbc_pct": 0.30,
        "lactate_pct": 0.25,
        "hgb_pct": -0.10,
        "hct_pct": -0.10,
        "fio2_pct": 0.10,
        "creat_pct": 0.0,
    },

    "Severe": {

        "temp_shift": 1.5,
        "hr_pct": 0.20,
        "rr_pct": 0.20,
        "wbc_pct": 0.50,
        "lactate_pct": 0.50,
        "hgb_pct": -0.20,
        "hct_pct": -0.20,
        "fio2_pct": 0.20,
        "creat_pct": 0.20,
    }
}


# ============================================================
# FEATURE FINDER
# ============================================================

def find_feature(
    feature_names,
    names
):

    feature_names = np.asarray(
        feature_names
    ).astype(str)

    for name in names:

        name_lower = name.lower()

        for i, feature in enumerate(
            feature_names
        ):

            if name_lower in feature.lower():
                return i

    raise ValueError(
        f"Could not find feature {names}"
    )


# ============================================================
# RAW SICU PERTURBATION
# ============================================================

def apply_sicu_perturbation_raw(
    X,
    icu,
    severity,
    feature_names
):

    X = list(X)

    icu = np.asarray(
        icu
    )

    feature_names = np.asarray(
        feature_names
    )

    params = SICU_PARAMS[
        severity
    ]

    hr_i = find_feature(
        feature_names,
        ["Heart Rate"]
    )

    temp_i = find_feature(
        feature_names,
        ["Temperature"]
    )

    rr_i = find_feature(
        feature_names,
        ["Respiratory rate"]
    )

    hgb_i = find_feature(
        feature_names,
        ["Hemoglobin"]
    )

    hct_i = find_feature(
        feature_names,
        ["Hematocrit"]
    )

    try:

        wbc_i = find_feature(
            feature_names,
            ["White blood cell count"]
        )

        has_wbc = True

    except ValueError:

        has_wbc = False

    try:

        lac_i = find_feature(
            feature_names,
            ["Lactate"]
        )

        has_lactate = True

    except ValueError:

        has_lactate = False

    try:

        fio2_i = find_feature(
            feature_names,
            ["fraction inspired oxygen"]
        )

        has_fio2 = True

    except ValueError:

        has_fio2 = False

    try:

        creat_i = find_feature(
            feature_names,
            ["Creatinine"]
        )

        has_creat = True

    except ValueError:

        has_creat = False

    for patient, unit in zip(
        X,
        icu
    ):

        if str(unit) != SICU_UNIT:
            continue

        for row in patient:

            # Temperature
            if row[temp_i] != "":

                v = safe_float(
                    row[temp_i]
                )

                if (
                    np.isfinite(v)
                    and
                    25 <= v <= 45
                ):

                    row[temp_i] = str(
                        np.clip(
                            v + params["temp_shift"],
                            25,
                            45
                        )
                    )

            # Heart rate
            if row[hr_i] != "":

                v = safe_float(
                    row[hr_i]
                )

                if (
                    np.isfinite(v)
                    and
                    20 <= v <= 250
                ):

                    row[hr_i] = str(
                        np.clip(
                            v * (
                                1 + params["hr_pct"]
                            ),
                            20,
                            250
                        )
                    )

            # Respiratory rate
            if row[rr_i] != "":

                v = safe_float(
                    row[rr_i]
                )

                if (
                    np.isfinite(v)
                    and
                    4 <= v <= 60
                ):

                    row[rr_i] = str(
                        np.clip(
                            v * (
                                1 + params["rr_pct"]
                            ),
                            4,
                            60
                        )
                    )

            # WBC
            if (
                has_wbc
                and params["wbc_pct"] > 0
                and row[wbc_i] != ""
            ):

                v = safe_float(
                    row[wbc_i]
                )

                if (
                    np.isfinite(v)
                    and
                    0 <= v <= 100
                ):

                    row[wbc_i] = str(
                        np.clip(
                            v * (
                                1 + params["wbc_pct"]
                            ),
                            0,
                            100
                        )
                    )

            # Lactate
            if (
                has_lactate
                and params["lactate_pct"] > 0
                and row[lac_i] != ""
            ):

                v = safe_float(
                    row[lac_i]
                )

                if (
                    np.isfinite(v)
                    and
                    0.1 <= v <= 30
                ):

                    row[lac_i] = str(
                        np.clip(
                            v * (
                                1 + params["lactate_pct"]
                            ),
                            0.1,
                            30
                        )
                    )

            # Hemoglobin
            if row[hgb_i] != "":

                v = safe_float(
                    row[hgb_i]
                )

                if (
                    np.isfinite(v)
                    and
                    2 <= v <= 25
                ):

                    row[hgb_i] = str(
                        np.clip(
                            v * (
                                1 + params["hgb_pct"]
                            ),
                            2,
                            25
                        )
                    )

            # Hematocrit
            if row[hct_i] != "":

                v = safe_float(
                    row[hct_i]
                )

                if (
                    np.isfinite(v)
                    and
                    5 <= v <= 65
                ):

                    row[hct_i] = str(
                        np.clip(
                            v * (
                                1 + params["hct_pct"]
                            ),
                            5,
                            65
                        )
                    )

            # FiO2
            if (
                has_fio2
                and params["fio2_pct"] > 0
                and row[fio2_i] != ""
            ):

                v = safe_float(
                    row[fio2_i]
                )

                if (
                    np.isfinite(v)
                    and
                    0.21 <= v <= 1
                ):

                    row[fio2_i] = str(
                        np.clip(
                            v * (
                                1 + params["fio2_pct"]
                            ),
                            0.21,
                            1.00
                        )
                    )

            # Creatinine
            if (
                has_creat
                and params["creat_pct"] > 0
                and row[creat_i] != ""
            ):

                v = safe_float(
                    row[creat_i]
                )

                if (
                    np.isfinite(v)
                    and
                    0.1 <= v <= 20
                ):

                    row[creat_i] = str(
                        np.clip(
                            v * (
                                1 + params["creat_pct"]
                            ),
                            0.1,
                            20
                        )
                    )

    return X


# ============================================================
# PIPELINE
# ============================================================

def load_pipeline(path):

    return joblib.load(
        path
    )


# ============================================================
# CLASSICAL TRANSFORMATION
# ============================================================

def transform_classical(
    X,
    icu,
    pipeline
):

    X = X[
        :,
        pipeline["non_empty_cols"]
    ]

    X = pipeline[
        "imputer"
    ].transform(
        X
    )

    X = pipeline[
        "variance_selector"
    ].transform(
        X
    )

    if len(
        pipeline["correlated_columns_removed"]
    ) > 0:

        drop = set(
            pipeline[
                "correlated_columns_removed"
            ]
        )

        mask = np.array([

            f not in drop

            for f in pipeline[
                "feature_names_after_variance"
            ]

        ])

        X = X[
            :,
            mask
        ]

    if (
        "scaler" in pipeline
        and
        pipeline["scaler"] is not None
    ):

        X = pipeline[
            "scaler"
        ].transform(
            X
        )

    icu_enc = pipeline[
        "icu_encoder"
    ].transform(
        np.asarray(
            icu
        ).reshape(-1, 1)
    )

    X = np.hstack([
        X,
        icu_enc
    ])

    return X


# ============================================================
# CLASSICAL FEATURE NAMES
# ============================================================

def get_classical_feature_names(
    pipeline
):

    if "final_feature_names" in pipeline:

        return np.asarray(
            pipeline["final_feature_names"]
        )

    feature_names = list(
        pipeline[
            "feature_names_after_variance"
        ]
    )

    if len(
        pipeline["correlated_columns_removed"]
    ) > 0:

        drop = set(
            pipeline[
                "correlated_columns_removed"
            ]
        )

        feature_names = [

            f for f in feature_names

            if f not in drop

        ]

    try:

        encoder = pipeline[
            "icu_encoder"
        ]

        icu_names = encoder.get_feature_names_out(
            ["ICU"]
        )

        feature_names.extend(
            icu_names.tolist()
        )

    except Exception:
        pass

    return np.asarray(
        feature_names
    )


# ============================================================
# MODEL UNWRAPPING
# ============================================================

def unwrap_calibrated_model(
    model
):

    if hasattr(
        model,
        "calibrated_classifiers_"
    ):

        calibrated = (
            model.calibrated_classifiers_
        )

        if len(calibrated) > 0:

            inner = calibrated[0]

            if hasattr(
                inner,
                "estimator"
            ):

                return inner.estimator

            if hasattr(
                inner,
                "base_estimator"
            ):

                return inner.base_estimator

    return model


# ============================================================
# PREDICTION WRAPPER
# ============================================================

def model_probability_function(
    model
):

    def predict(X):

        p = model.predict_proba(
            X
        )

        return p[:, 1]

    return predict


# ============================================================
# SHAP VALUE NORMALIZATION
# ============================================================

def normalize_shap_values(
    shap_values
):

    if isinstance(
        shap_values,
        list
    ):

        if len(shap_values) == 2:

            shap_values = shap_values[1]

        else:

            shap_values = shap_values[0]

    return np.asarray(
        shap_values
    )


# ============================================================
# CLASSICAL FEATURE IMPORTANCE
# ============================================================

def mean_abs_shap(
    shap_values,
    feature_names
):

    shap_values = normalize_shap_values(
        shap_values
    )

    if shap_values.ndim == 2:

        values = np.mean(
            np.abs(
                shap_values
            ),
            axis=0
        )

    elif shap_values.ndim == 3:

        if shap_values.shape[2] == 2:

            shap_positive = (
                shap_values[:, :, 1]
            )

            values = np.mean(
                np.abs(
                    shap_positive
                ),
                axis=0
            )

        else:

            raise ValueError(
                "Unsupported 3D SHAP shape: "
                f"{shap_values.shape}"
            )

    else:

        raise ValueError(
            "Unsupported SHAP shape: "
            f"{shap_values.shape}"
        )

    feature_names = np.asarray(
        feature_names
    )

    if len(values) != len(feature_names):

        raise ValueError(
            "SHAP feature count does not "
            "match feature-name count: "
            f"{len(values)} vs "
            f"{len(feature_names)}"
        )

    df = pd.DataFrame({

        "Feature":
            feature_names,

        "Mean_Abs_SHAP":
            values

    })

    df = df.sort_values(
        "Mean_Abs_SHAP",
        ascending=False
    ).reset_index(
        drop=True
    )

    df["Rank"] = np.arange(
        1,
        len(df) + 1
    )

    return df


# ============================================================
# DEEP FEATURE IMPORTANCE
# ============================================================

def mean_abs_shap_deep(
    shap_values,
    feature_names
):

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim != 3:

        raise ValueError(
            "Deep SHAP values must have "
            "shape (N, T, F). Got "
            f"{shap_values.shape}"
        )

    values = np.mean(
        np.abs(
            shap_values
        ),
        axis=(0, 1)
    )

    feature_names = np.asarray(
        feature_names
    )

    if len(values) != len(feature_names):

        raise ValueError(
            "Deep SHAP feature count does not "
            "match feature-name count: "
            f"{len(values)} vs "
            f"{len(feature_names)}"
        )

    df = pd.DataFrame({

        "Feature":
            feature_names,

        "Mean_Abs_SHAP":
            values

    })

    df = df.sort_values(
        "Mean_Abs_SHAP",
        ascending=False
    ).reset_index(
        drop=True
    )

    df["Rank"] = np.arange(
        1,
        len(df) + 1
    )

    return df


# ============================================================
# PRINT SHAP RESULTS
#
# Mirrors MICU_SHAP.py exactly so console output and the
# Top-10 overlap figure are computed and reported identically
# across both scripts.
# ============================================================

def print_shap_results(
    model_name,
    comparison,
    reference_df,
    perturbed_df,
    rho,
    p_value
):

    print()
    print(
        "============================================================"
    )

    print(
        f"{model_name}: "
        f"Spearman rho = {rho:.4f}, "
        f"p = {p_value:.4g}"
    )

    # --------------------------------------------------------
    # Top-10 rows by absolute SHAP change
    # --------------------------------------------------------

    print()
    print(
        "Top 10 rows "
        "(by absolute SHAP change):"
    )

    top10_display = comparison.head(10)[
        [
            "Feature",
            "Mean_Abs_SHAP_Reference",
            "Mean_Abs_SHAP_Perturbed",
            "SHAP_Change",
            "Rank_Reference",
            "Rank_Perturbed",
            "Rank_Change"
        ]
    ]

    print(
        top10_display.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Top-10 overlap
    # --------------------------------------------------------

    top10_reference = set(
        reference_df.head(10)["Feature"]
    )

    top10_perturbed = set(
        perturbed_df.head(10)["Feature"]
    )

    overlap = (
        top10_reference
        &
        top10_perturbed
    )

    print(
        f"Top-10 feature overlap: "
        f"{len(overlap)}/10"
    )

    # --------------------------------------------------------
    # Top features increasing
    # --------------------------------------------------------

    increasing = comparison[
        comparison["SHAP_Change"] > 0
    ].sort_values(
        "SHAP_Change",
        ascending=False
    ).head(10)

    print()
    print(
        "Top features increasing under perturbation:"
    )

    if len(increasing) == 0:

        print(
            "None"
        )

    else:

        increasing_display = (
            increasing[
                [
                    "Feature",
                    "Mean_Abs_SHAP_Reference",
                    "Mean_Abs_SHAP_Perturbed"
                ]
            ]
            .rename(columns={
                "Mean_Abs_SHAP_Reference":
                    "Reference",

                "Mean_Abs_SHAP_Perturbed":
                    "Perturbed"
            })
        )

        print(
            increasing_display.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Rank comparison
    # --------------------------------------------------------

    print()
    print(
        "Feature rank comparison:"
    )

    rank_display = (

        comparison.sort_values(
            "Rank_Reference"
        )[[
            "Feature",
            "Rank_Reference",
            "Rank_Perturbed"
        ]]

    )

    print(
        rank_display.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SHAP change
    # --------------------------------------------------------

    print()
    print(
        "Feature SHAP change:"
    )

    change_display = (

        comparison.sort_values(
            "SHAP_Change",
            ascending=False
        )[[
            "Feature",
            "SHAP_Change"
        ]]

    )

    print(
        change_display.to_string(
            index=False
        )
    )


# ============================================================
# SHAP COMPARISON
#
# SAME LOGIC FOR ALL 8 MODELS, matching MICU_SHAP.py's
# compare_shap exactly (including its return signature) so
# that the Top-10 overlap figure can be reused downstream
# from in-memory reference/perturbed feature-importance
# tables rather than re-derived from disk.
# ============================================================

def compare_shap(
    reference_shap,
    perturbed_shap,
    feature_names,
    out_dir,
    suffix,
    deep=False
):

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    if deep:

        reference_df = mean_abs_shap_deep(
            reference_shap,
            feature_names
        )

        perturbed_df = mean_abs_shap_deep(
            perturbed_shap,
            feature_names
        )

    else:

        reference_df = mean_abs_shap(
            reference_shap,
            feature_names
        )

        perturbed_df = mean_abs_shap(
            perturbed_shap,
            feature_names
        )

    # ========================================================
    # SAVE INDIVIDUAL FEATURE IMPORTANCE
    # ========================================================

    reference_df.to_csv(

        os.path.join(
            out_dir,
            f"mean_abs_shap_reference_{suffix}.csv"
        ),

        index=False
    )

    perturbed_df.to_csv(

        os.path.join(
            out_dir,
            f"mean_abs_shap_perturbed_{suffix}.csv"
        ),

        index=False
    )

    # ========================================================
    # MERGE
    # ========================================================

    comparison = pd.merge(

        reference_df[
            [
                "Feature",
                "Mean_Abs_SHAP",
                "Rank"
            ]
        ],

        perturbed_df[
            [
                "Feature",
                "Mean_Abs_SHAP",
                "Rank"
            ]
        ],

        on="Feature",

        suffixes=(
            "_Reference",
            "_Perturbed"
        )
    )

    # ========================================================
    # SHAP CHANGE
    # ========================================================

    comparison[
        "SHAP_Change"
    ] = (

        comparison[
            "Mean_Abs_SHAP_Perturbed"
        ]

        -

        comparison[
            "Mean_Abs_SHAP_Reference"
        ]
    )

    comparison[
        "Absolute_SHAP_Change"
    ] = np.abs(

        comparison[
            "SHAP_Change"
        ]
    )

    comparison[
        "Rank_Change"
    ] = (

        comparison[
            "Rank_Perturbed"
        ]

        -

        comparison[
            "Rank_Reference"
        ]
    )

    # ========================================================
    # SAVE FULL COMPARISON
    # ========================================================

    comparison = comparison.sort_values(

        "Absolute_SHAP_Change",

        ascending=False
    ).reset_index(
        drop=True
    )

    comparison.to_csv(

        os.path.join(
            out_dir,
            f"shap_comparison_{suffix}.csv"
        ),

        index=False
    )

    # ========================================================
    # SPEARMAN
    # ========================================================

    merged_rank = comparison.sort_values(
        "Feature"
    )

    rho, p_value = spearmanr(

        merged_rank[
            "Rank_Reference"
        ],

        merged_rank[
            "Rank_Perturbed"
        ]
    )

    pd.DataFrame({

        "Spearman_Rho": [rho],

        "P_Value": [p_value]

    }).to_csv(

        os.path.join(
            out_dir,
            f"rank_correlation_{suffix}.csv"
        ),

        index=False
    )

    # ========================================================
    # TOP-10 OVERLAP
    # ========================================================

    top10_reference = set(
        reference_df.head(10)["Feature"]
    )

    top10_perturbed = set(
        perturbed_df.head(10)["Feature"]
    )

    overlap = (
        top10_reference
        &
        top10_perturbed
    )

    overlap_count = len(
        overlap
    )

    overlap_fraction = (
        overlap_count / 10
    )

    pd.DataFrame({

        "Reference_Top10": [

            ", ".join(
                reference_df.head(10)["Feature"]
            )

        ],

        "Perturbed_Top10": [

            ", ".join(
                perturbed_df.head(10)["Feature"]
            )

        ],

        "Overlap_Count": [
            overlap_count
        ],

        "Overlap_Fraction": [
            overlap_fraction
        ]

    }).to_csv(

        os.path.join(
            out_dir,
            f"top10_overlap_{suffix}.csv"
        ),

        index=False
    )

    # ========================================================
    # TOP FEATURES INCREASING (still saved to CSV, unchanged)
    # ========================================================

    increasing = comparison[
        comparison[
            "SHAP_Change"
        ] > 0
    ].copy()

    increasing = increasing.sort_values(

        "SHAP_Change",

        ascending=False
    ).reset_index(
        drop=True
    )

    top_increasing = increasing.head(
        10
    )

    top_increasing[
        [
            "Feature",
            "Mean_Abs_SHAP_Reference",
            "Mean_Abs_SHAP_Perturbed",
            "Rank_Reference",
            "Rank_Perturbed",
            "SHAP_Change"
        ]
    ].to_csv(

        os.path.join(
            out_dir,
            f"top_increasing_features_{suffix}.csv"
        ),

        index=False
    )

    # ========================================================
    # SHAP CHANGE PLOT
    # ========================================================

    plot_df = comparison.head(

        min(
            15,
            len(comparison)
        )

    ).copy()

    plot_df = plot_df.sort_values(
        "SHAP_Change"
    )

    plt.figure(
        figsize=(9, 7)
    )

    plt.barh(

        plot_df["Feature"],

        plot_df["SHAP_Change"]
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel(
        "Change in Mean |SHAP|"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Feature Attribution Change\n"
        "Baseline → SICU Perturbation"
    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            out_dir,
            f"shap_change_{suffix}.png"
        ),

        dpi=300
    )

    plt.close()

    # ========================================================
    # RANK SCATTER
    # ========================================================

    plt.figure(
        figsize=(7, 7)
    )

    plt.scatter(

        comparison[
            "Rank_Reference"
        ],

        comparison[
            "Rank_Perturbed"
        ],

        alpha=0.7
    )

    lim = len(
        feature_names
    )

    plt.plot(
        [1, lim],
        [1, lim],
        linestyle="--"
    )

    plt.xlabel(
        "Baseline SHAP Rank"
    )

    plt.ylabel(
        "Perturbed SHAP Rank"
    )

    plt.title(
        f"SHAP Rank Stability\n"
        f"Spearman ρ = {rho:.3f}"
    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            out_dir,
            f"rank_scatter_{suffix}.png"
        ),

        dpi=300
    )

    plt.close()

    # ========================================================
    # CONSOLE OUTPUT — identical to MICU_SHAP.py
    # ========================================================

    print_shap_results(

        model_name=suffix,

        comparison=comparison,

        reference_df=reference_df,

        perturbed_df=perturbed_df,

        rho=rho,

        p_value=p_value
    )

    return (
        comparison,
        rho,
        p_value,
        reference_df,
        perturbed_df
    )


# ============================================================
# CLASSICAL SHAP
# ============================================================

def run_classical_shap(
    model_name,
    pipeline_path,
    X_reference,
    X_perturbed,
    icu_reference,
    icu_perturbed,
    out_dir
):

    print(
        f"\nRunning SHAP: {model_name}"
    )

    pipeline = load_pipeline(
        pipeline_path
    )

    model = pipeline[
        "model"
    ]

    X_ref = transform_classical(
        X_reference,
        icu_reference,
        pipeline
    )

    X_pert = transform_classical(
        X_perturbed,
        icu_perturbed,
        pipeline
    )

    feature_names = get_classical_feature_names(
        pipeline
    )

    if len(feature_names) != X_ref.shape[1]:

        raise RuntimeError(
            f"{model_name}: feature-name count "
            f"{len(feature_names)} does not match "
            f"transformed feature count "
            f"{X_ref.shape[1]}"
        )

    explain_model = unwrap_calibrated_model(
        model
    )

    # ========================================================
    # XGBoost
    # ========================================================

    if model_name == "XGBoost":

        explainer = shap.TreeExplainer(
            explain_model
        )

        shap_ref = normalize_shap_values(
            explainer.shap_values(
                X_ref
            )
        )

        shap_pert = normalize_shap_values(
            explainer.shap_values(
                X_pert
            )
        )

    # ========================================================
    # Logistic Regression
    # ========================================================

    elif model_name == "LogisticRegression":

        background_size = min(
            100,
            len(X_ref)
        )

        rng = np.random.default_rng(
            RANDOM_SEED
        )

        background_idx = rng.choice(

            len(X_ref),

            size=background_size,

            replace=False
        )

        background = X_ref[
            background_idx
        ]

        explainer = shap.LinearExplainer(

            explain_model,

            background
        )

        shap_ref = normalize_shap_values(
            explainer.shap_values(
                X_ref
            )
        )

        shap_pert = normalize_shap_values(
            explainer.shap_values(
                X_pert
            )
        )

    # ========================================================
    # Random Forest
    # ========================================================

    elif model_name == "Random Forest":

        background_size = min(
            100,
            len(X_ref)
        )

        rng = np.random.default_rng(
            RANDOM_SEED
        )

        background_idx = rng.choice(

            len(X_ref),

            size=background_size,

            replace=False
        )

        background = X_ref[
            background_idx
        ]

        explainer = shap.TreeExplainer(

            explain_model,

            data=background
        )

        shap_ref = normalize_shap_values(
            explainer.shap_values(
                X_ref
            )
        )

        shap_pert = normalize_shap_values(
            explainer.shap_values(
                X_pert
            )
        )

    # ========================================================
    # MLP
    # ========================================================

    elif model_name == "MLP":

        background_size = min(
            100,
            len(X_ref)
        )

        rng = np.random.default_rng(
            RANDOM_SEED
        )

        background_idx = rng.choice(

            len(X_ref),

            size=background_size,

            replace=False
        )

        background = X_ref[
            background_idx
        ]

        explainer = shap.KernelExplainer(

            model_probability_function(
                explain_model
            ),

            background
        )

        n_explain = min(
            100,
            len(X_ref)
        )

        ref_idx = np.arange(
            n_explain
        )

        pert_idx = np.arange(
            n_explain
        )

        shap_ref = normalize_shap_values(

            explainer.shap_values(

                X_ref[ref_idx],

                silent=True
            )
        )

        shap_pert = normalize_shap_values(

            explainer.shap_values(

                X_pert[pert_idx],

                silent=True
            )
        )

    else:

        raise ValueError(
            f"Unsupported classical model: "
            f"{model_name}"
        )

    # ========================================================
    # SAVE RAW SHAP
    # ========================================================

    np.save(

        os.path.join(
            out_dir,
            f"shap_reference_{model_name}.npy"
        ),

        shap_ref
    )

    np.save(

        os.path.join(
            out_dir,
            f"shap_perturbed_{model_name}.npy"
        ),

        shap_pert
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    suffix = model_name.lower().replace(
        " ",
        "_"
    )

    (
        comparison,
        rho,
        p,
        reference_df,
        perturbed_df
    ) = compare_shap(

        shap_ref,
        shap_pert,

        feature_names,

        out_dir,

        suffix,

        deep=False
    )

    return {
        "comparison": comparison,
        "rho": rho,
        "p": p,
        "reference_df": reference_df,
        "perturbed_df": perturbed_df
    }


# ============================================================
# INVERSE NORMALIZE
# ============================================================

def inverse_normalize(
    X,
    normalizer,
    cont_channels
):

    X = np.asarray(
        X,
        dtype=np.float32
    ).copy()

    means = np.asarray(
        normalizer._means
    )

    stds = np.asarray(
        normalizer._stds
    )

    for ch in cont_channels:

        valid = X[:, :, ch] != 0

        X[
            valid,
            ch
        ] = (
            X[
                valid,
                ch
            ]
            *
            stds[ch]
            +
            means[ch]
        )

    return X


# ============================================================
# RENORMALIZE
# ============================================================

def renormalize(
    X,
    normalizer,
    cont_channels
):

    X = np.asarray(
        X,
        dtype=np.float32
    ).copy()

    means = np.asarray(
        normalizer._means
    )

    stds = np.asarray(
        normalizer._stds
    )

    for ch in cont_channels:

        valid = X[:, :, ch] != 0

        X[
            valid,
            ch
        ] = (
            X[
                valid,
                ch
            ]
            -
            means[ch]
        ) / stds[ch]

    return X


# ============================================================
# DEEP SICU PERTURBATION
# ============================================================

def apply_sicu_perturbation_tensor(
    X,
    icu,
    severity,
    feature_names,
    normalizer,
    cont_channels
):

    X = np.asarray(
        X,
        dtype=np.float32
    ).copy()

    if X.ndim != 3:

        raise RuntimeError(
            "Expected tensor shape "
            "(N,T,F). Got "
            f"{X.shape}"
        )

    icu = np.asarray(
        icu
    )

    feature_names = np.asarray(
        feature_names
    )

    if len(icu) != len(X):

        raise RuntimeError(
            "ICU labels do not match "
            "number of patients."
        )

    sicu_mask = np.array([

        str(x) == SICU_UNIT

        for x in icu

    ])

    if sicu_mask.sum() == 0:

        return X

    X_raw = inverse_normalize(

        X,

        normalizer,

        cont_channels
    )

    params = SICU_PARAMS[
        severity
    ]

    # ========================================================
    # FIND FEATURES
    # ========================================================

    hr_i = find_feature(
        feature_names,
        ["Heart Rate"]
    )

    temp_i = find_feature(
        feature_names,
        ["Temperature"]
    )

    rr_i = find_feature(
        feature_names,
        ["Respiratory rate"]
    )

    hgb_i = find_feature(
        feature_names,
        ["Hemoglobin"]
    )

    hct_i = find_feature(
        feature_names,
        ["Hematocrit"]
    )

    try:

        wbc_i = find_feature(
            feature_names,
            ["White blood cell count"]
        )

        has_wbc = True

    except ValueError:

        has_wbc = False

    try:

        lac_i = find_feature(
            feature_names,
            ["Lactate"]
        )

        has_lactate = True

    except ValueError:

        has_lactate = False

    try:

        fio2_i = find_feature(
            feature_names,
            ["fraction inspired oxygen"]
        )

        has_fio2 = True

    except ValueError:

        has_fio2 = False

    try:

        creat_i = find_feature(
            feature_names,
            ["Creatinine"]
        )

        has_creat = True

    except ValueError:

        has_creat = False

    # ========================================================
    # PERCENTAGE PERTURBATION
    # ========================================================

    def perturb_pct(
        arr,
        pct,
        lo,
        hi
    ):

        valid = arr != 0

        arr[valid] = np.clip(

            arr[valid]
            *
            (1 + pct),

            lo,
            hi
        )

        return arr

    # ========================================================
    # TEMPERATURE
    # ========================================================

    temp = X_raw[
        sicu_mask,
        :,
        temp_i
    ]

    valid = temp != 0

    temp[valid] = np.clip(

        temp[valid]
        +
        params["temp_shift"],

        25,
        45
    )

    X_raw[
        sicu_mask,
        :,
        temp_i
    ] = temp

    # ========================================================
    # HEART RATE
    # ========================================================

    X_raw[
        sicu_mask,
        :,
        hr_i
    ] = perturb_pct(

        X_raw[
            sicu_mask,
            :,
            hr_i
        ],

        params["hr_pct"],

        20,

        250
    )

    # ========================================================
    # RESPIRATORY RATE
    # ========================================================

    X_raw[
        sicu_mask,
        :,
        rr_i
    ] = perturb_pct(

        X_raw[
            sicu_mask,
            :,
            rr_i
        ],

        params["rr_pct"],

        4,

        60
    )

    # ========================================================
    # WBC
    # ========================================================

    if (
        has_wbc
        and
        params["wbc_pct"] > 0
    ):

        X_raw[
            sicu_mask,
            :,
            wbc_i
        ] = perturb_pct(

            X_raw[
                sicu_mask,
                :,
                wbc_i
            ],

            params["wbc_pct"],

            0,

            100
        )

    # ========================================================
    # LACTATE
    # ========================================================

    if (
        has_lactate
        and
        params["lactate_pct"] > 0
    ):

        X_raw[
            sicu_mask,
            :,
            lac_i
        ] = perturb_pct(

            X_raw[
                sicu_mask,
                :,
                lac_i
            ],

            params["lactate_pct"],

            0.1,

            30
        )

    # ========================================================
    # HEMOGLOBIN
    # ========================================================

    X_raw[
        sicu_mask,
        :,
        hgb_i
    ] = perturb_pct(

        X_raw[
            sicu_mask,
            :,
            hgb_i
        ],

        params["hgb_pct"],

        2,

        25
    )

    # ========================================================
    # HEMATOCRIT
    # ========================================================

    X_raw[
        sicu_mask,
        :,
        hct_i
    ] = perturb_pct(

        X_raw[
            sicu_mask,
            :,
            hct_i
        ],

        params["hct_pct"],

        5,

        65
    )

    # ========================================================
    # FIO2
    # ========================================================

    if (
        has_fio2
        and
        params["fio2_pct"] > 0
    ):

        X_raw[
            sicu_mask,
            :,
            fio2_i
        ] = perturb_pct(

            X_raw[
                sicu_mask,
                :,
                fio2_i
            ],

            params["fio2_pct"],

            0.21,

            1.00
        )

    # ========================================================
    # CREATININE
    # ========================================================

    if (
        has_creat
        and
        params["creat_pct"] > 0
    ):

        X_raw[
            sicu_mask,
            :,
            creat_i
        ] = perturb_pct(

            X_raw[
                sicu_mask,
                :,
                creat_i
            ],

            params["creat_pct"],

            0.1,

            20
        )

    # ========================================================
    # RE-NORMALIZE
    # ========================================================

    X_perturbed = renormalize(

        X_raw,

        normalizer,

        cont_channels
    )

    return X_perturbed


# ============================================================
# LOAD STANDARD LSTM MODULE
# ============================================================

def load_lstm_module():

    module_path = os.path.join(

        BASE_DIR,

        "mimic4models",

        "keras_models",

        "lstm.py"
    )

    spec = importlib.util.spec_from_file_location(

        "mimic_lstm",

        module_path
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# LOAD CHANNELWISE LSTM MODULE
# ============================================================

def load_channelwise_lstm_module():

    # --------------------------------------------------------
    # The actual filename on disk (confirmed working via
    # MICU_SHAP.py) is "channel_wise_lstms.py" — plural.
    # A couple of singular variants are also checked as a
    # fallback in case the filename differs across checkouts.
    # --------------------------------------------------------

    candidate_names = [

        "channel_wise_lstms.py",

        "channel_wise_lstm.py",

        "channelwise_lstm.py",

        "channelwise_lstms.py",
    ]

    candidate_paths = [

        os.path.join(

            BASE_DIR,

            "mimic4models",

            "keras_models",

            name
        )

        for name in candidate_names

    ]

    module_path = next(

        (p for p in candidate_paths if os.path.exists(p)),

        None
    )

    if module_path is None:

        checked = "\n".join(candidate_paths)

        raise FileNotFoundError(

            "Could not find channel-wise LSTM module. "
            f"Checked:\n{checked}"
        )

    spec = importlib.util.spec_from_file_location(

        "mimic_channelwise_lstm",

        module_path
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# FIND DEEP MODEL WEIGHTS
#
# This avoids hard-coding filenames for the three models whose
# exact filenames were not included in the original script.
# ============================================================

def find_deep_model_weights(
    model_name
):

    # --------------------------------------------------------
    # Two possible layouts are used across models:
    #
    #   in_hospital_mortality/{model_name}/keras_states
    #       (LSTM_DS, ChannelwiseLSTM, ChannelwiseLSTM_DS)
    #
    #   in_hospital_mortality/keras_states
    #       (plain LSTM has no per-model subfolder)
    #
    # Try the model-specific subfolder first, then fall back
    # to the shared keras_states directory.
    # --------------------------------------------------------

    candidate_dirs = [

        os.path.join(

            BASE_DIR,

            "mimic4models",

            "in_hospital_mortality",

            model_name,

            "keras_states"
        ),

        os.path.join(

            BASE_DIR,

            "mimic4models",

            "in_hospital_mortality",

            "keras_states"
        ),
    ]

    states_dir = next(

        (d for d in candidate_dirs if os.path.exists(d)),

        None
    )

    if states_dir is None:

        checked = "\n".join(candidate_dirs)

        raise FileNotFoundError(

            f"Could not find a keras_states directory for "
            f"{model_name}. Checked:\n{checked}"
        )

    files = sorted(

        glob.glob(
            os.path.join(
                states_dir,
                "*.keras"
            )
        )
    )

    if len(files) == 0:

        files = sorted(

            glob.glob(
                os.path.join(
                    states_dir,
                    "*.h5"
                )
            )
        )

    if len(files) == 0:

        files = sorted(

            glob.glob(
                os.path.join(
                    states_dir,
                    "*.hdf5"
                )
            )
        )

    if len(files) == 0:

        raise FileNotFoundError(

            f"No Keras model weights found for "
            f"{model_name} in:\n{states_dir}"
        )

    # --------------------------------------------------------
    # Prefer files containing the expected architecture.
    # --------------------------------------------------------

    preferred = [

        f for f in files

        if (
            "n16" in os.path.basename(f).lower()
            and
            "d0.3" in os.path.basename(f).lower()
            and
            "dep2" in os.path.basename(f).lower()
        )

    ]

    if len(preferred) > 0:

        files = preferred

    # --------------------------------------------------------
    # When falling back to a shared keras_states directory
    # (the plain-LSTM case), the same folder can also contain
    # ChannelwiseLSTM or *_DS checkpoints. Filter by model
    # family and target-replication (DS) status so the wrong
    # architecture is never silently selected.
    # --------------------------------------------------------

    is_channelwise = "channelwise" in model_name.lower()

    is_ds = model_name.endswith("_DS")

    family_filtered = [

        f for f in files

        if (
            (
                ("channel_wise" in os.path.basename(f).lower()
                 or "channelwise" in os.path.basename(f).lower())
                if is_channelwise else
                ("k_lstm" in os.path.basename(f).lower())
            )
            and
            (
                ("trc" in os.path.basename(f).lower())
                if is_ds else
                ("trc" not in os.path.basename(f).lower())
            )
        )

    ]

    if len(family_filtered) > 0:

        files = family_filtered

    # --------------------------------------------------------
    # Prefer latest epoch if filenames contain epoch numbers.
    # Otherwise use the first deterministic file.
    # --------------------------------------------------------

    files = sorted(
        files
    )

    selected = files[-1]

    print()
    print(
        f"{model_name} weights:"
    )
    print(
        selected
    )

    return selected


# ============================================================
# BUILD DEEP MODEL
#
# The four architectures use the same hyperparameters as the
# benchmark models used in the thesis.
# ============================================================

def build_deep_model(
    model_name,
    input_dim,
    header=None
):

    if model_name in [
        "LSTM",
        "LSTM_DS"
    ]:

        lstm_module = load_lstm_module()

        target_repl = (
            model_name == "LSTM_DS"
        )

        model = lstm_module.Network(

            dim=16,

            batch_norm=False,

            dropout=0.3,

            rec_dropout=0.0,

            task="ihm",

            target_repl=target_repl,

            deep_supervision=False,

            num_classes=1,

            depth=2,

            input_dim=input_dim
        )

        return model

    elif model_name in [
        "ChannelwiseLSTM",
        "ChannelwiseLSTM_DS"
    ]:

        channelwise_module = (
            load_channelwise_lstm_module()
        )

        target_repl = (
            model_name == "ChannelwiseLSTM_DS"
        )

        if header is None:

            raise ValueError(

                "ChannelwiseLSTM/ChannelwiseLSTM_DS "
                "require a 'header' (feature name list) "
                "to build the per-channel sub-networks."
            )

        # ----------------------------------------------------
        # Matches MICU_SHAP.py's working call exactly: no
        # num_classes argument for this architecture, and
        # header= is required by the Network constructor.
        # ----------------------------------------------------

        return channelwise_module.Network(

            dim=16,

            batch_norm=False,

            dropout=0.3,

            rec_dropout=0.0,

            header=header,

            task="ihm",

            target_repl=target_repl,

            deep_supervision=False,

            depth=2,

            input_dim=input_dim,

            size_coef=4.0
        )

    else:

        raise ValueError(
            f"Unsupported deep model: {model_name}"
        )


# ============================================================
# GET MORTALITY OUTPUT
# ============================================================

def get_mortality_output(
    outputs
):

    if isinstance(
        outputs,
        (list, tuple)
    ):

        mortality_output = outputs[0]

    else:

        mortality_output = outputs

    if len(
        mortality_output.shape
    ) != 2:

        raise ValueError(

            "Mortality output must be "
            "(batch, output). Got "
            f"{mortality_output.shape}"
        )

    if mortality_output.shape[-1] != 1:

        raise ValueError(

            "Expected one mortality output. Got "
            f"{mortality_output.shape}"
        )

    return mortality_output


# ============================================================
# PROCESS DEEP SHAP
# ============================================================

def process_deep_shap(
    values,
    n_timesteps,
    n_features
):

    if isinstance(
        values,
        list
    ):

        if len(values) != 1:

            raise ValueError(
                "Expected exactly one SHAP output."
            )

        values = values[0]

    values = np.asarray(
        values
    )

    # --------------------------------------------------------
    # SHAP may return:
    #
    # (N,T,F,1)
    #
    # or:
    #
    # (N,T,F)
    # --------------------------------------------------------

    if (
        values.ndim == 4
        and
        values.shape[-1] == 1
    ):

        values = np.squeeze(
            values,
            axis=-1
        )

    if values.ndim != 3:

        raise ValueError(

            "Expected deep SHAP shape "
            "(N,T,F). Got "
            f"{values.shape}"
        )

    expected_shape = (

        values.shape[0],

        n_timesteps,

        n_features
    )

    if values.shape != expected_shape:

        raise ValueError(

            "Unexpected deep SHAP shape. "
            f"Expected {expected_shape}, "
            f"got {values.shape}"
        )

    return values


# ============================================================
# GENERIC DEEP SHAP
#
# Used for:
#
# LSTM
# LSTM_DS
# ChannelwiseLSTM
# ChannelwiseLSTM_DS
# ============================================================

def run_deep_shap(
    model_name,
    model_path,
    pipeline_path,
    X_reference,
    X_perturbed,
    out_dir,
    severity
):

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    print()
    print(
        "=========================================="
    )
    print(
        f"Running deep SHAP: {model_name}"
    )
    print(
        "=========================================="
    )

    # ========================================================
    # LOAD PIPELINE
    # ========================================================

    pipeline = joblib.load(
        pipeline_path
    )

    feature_names = list(
        pipeline[
            "feature_names"
        ]
    )

    # ========================================================
    # INPUT ARRAYS
    # ========================================================

    X_reference = np.asarray(
        X_reference,
        dtype=np.float32
    )

    X_perturbed = np.asarray(
        X_perturbed,
        dtype=np.float32
    )

    if X_reference.ndim != 3:

        raise ValueError(

            "X_reference must have shape "
            "(N,T,F). Got "
            f"{X_reference.shape}"
        )

    if X_perturbed.ndim != 3:

        raise ValueError(

            "X_perturbed must have shape "
            "(N,T,F). Got "
            f"{X_perturbed.shape}"
        )

    if X_reference.shape != X_perturbed.shape:

        raise ValueError(

            "Reference and perturbed tensors "
            "must have identical shapes. "
            f"Reference: {X_reference.shape}, "
            f"Perturbed: {X_perturbed.shape}"
        )

    n_timesteps = X_reference.shape[1]

    n_features = X_reference.shape[2]

    if len(feature_names) != n_features:

        raise ValueError(

            f"{model_name}: number of feature names "
            f"({len(feature_names)}) does not match "
            f"input features ({n_features})."
        )

    print(
        "Deep SHAP input shape:",
        X_reference.shape
    )

    # ========================================================
    # BUILD MODEL
    # ========================================================

    model = build_deep_model(

        model_name,

        n_features,

        header=feature_names
    )

    # ========================================================
    # LOAD WEIGHTS
    # ========================================================

    print(
        "\nLoading weights:"
    )

    print(
        model_path
    )

    model.load_weights(
        model_path
    )

    # ========================================================
    # SAMPLING
    #
    # Paired patient indices are used so that:
    #
    # reference[i]
    #
    # is compared against:
    #
    # perturbed[i]
    #
    # for the same patient.
    # ========================================================

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    n_background = min(
        50,
        len(X_reference)
    )

    n_explain = min(
        100,
        len(X_reference)
    )

    background_idx = rng.choice(

        len(X_reference),

        size=n_background,

        replace=False
    )

    explain_idx = rng.choice(

        len(X_reference),

        size=n_explain,

        replace=False
    )

    background = X_reference[
        background_idx
    ]

    reference_sample = X_reference[
        explain_idx
    ]

    perturbed_sample = X_perturbed[
        explain_idx
    ]

    print(
        "SHAP background:",
        background.shape
    )

    print(
        "SHAP reference:",
        reference_sample.shape
    )

    print(
        "SHAP perturbed:",
        perturbed_sample.shape
    )

    # ========================================================
    # FIXED-SHAPE KERAS INPUT
    # ========================================================

    fixed_input = tf.keras.Input(

        shape=(

            n_timesteps,

            n_features

        ),

        dtype=tf.float32,

        name="shap_input"
    )

    outputs = model(
        fixed_input,
        training=False
    )

    mortality_output = get_mortality_output(
        outputs
    )

    # ========================================================
    # FIXED KERAS MODEL
    # ========================================================

    fixed_model = tf.keras.Model(

        inputs=fixed_input,

        outputs=mortality_output,

        name=f"{model_name}_mortality_SHAP"
    )

    print(
        "Fixed model input shape:",
        fixed_model.input_shape
    )

    print(
        "Fixed model output shape:",
        fixed_model.output_shape
    )

    # ========================================================
    # VALIDATE PREDICTION
    # ========================================================

    test_prediction = fixed_model.predict(

        background[:2],

        verbose=0
    )

    if test_prediction.shape != (
        2,
        1
    ):

        raise ValueError(

            f"{model_name}: wrapped model does not "
            "produce expected (N,1) output. Got "
            f"{test_prediction.shape}"
        )

    # ========================================================
    # GRADIENT SHAP
    # ========================================================

    print(
        "\nCreating GradientExplainer..."
    )

    explainer = shap.GradientExplainer(

        (
            fixed_input,
            mortality_output
        ),

        background
    )

    print(
        "GradientExplainer created successfully."
    )

    # ========================================================
    # REFERENCE SHAP
    # ========================================================

    print(
        "\nCalculating reference SHAP..."
    )

    shap_reference_raw = (
        explainer.shap_values(

            reference_sample,

            nsamples=200
        )
    )

    # ========================================================
    # PERTURBED SHAP
    # ========================================================

    print(
        "Calculating perturbed SHAP..."
    )

    shap_perturbed_raw = (
        explainer.shap_values(

            perturbed_sample,

            nsamples=200
        )
    )

    # ========================================================
    # NORMALIZE SHAP
    # ========================================================

    shap_reference = process_deep_shap(

        shap_reference_raw,

        n_timesteps,

        n_features
    )

    shap_perturbed = process_deep_shap(

        shap_perturbed_raw,

        n_timesteps,

        n_features
    )

    print(
        "Reference SHAP shape:",
        shap_reference.shape
    )

    print(
        "Perturbed SHAP shape:",
        shap_perturbed.shape
    )

    # ========================================================
    # MODEL SUFFIX
    # ========================================================

    suffix = (

        model_name
        .lower()
        .replace(" ", "_")
        +
        f"_{severity.lower()}"
    )

    # ========================================================
    # SAVE RAW SHAP
    # ========================================================

    np.save(

        os.path.join(
            out_dir,
            f"shap_reference_{suffix}.npy"
        ),

        shap_reference
    )

    np.save(

        os.path.join(
            out_dir,
            f"shap_perturbed_{suffix}.npy"
        ),

        shap_perturbed
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    (
        comparison,
        rho,
        p,
        reference_df,
        perturbed_df
    ) = compare_shap(

        shap_reference,

        shap_perturbed,

        feature_names,

        out_dir,

        suffix,

        deep=True
    )

    # ========================================================
    # TEMPORAL AGGREGATION
    #
    # ONLY used for summary plots.
    #
    # The ranking and overlap calculations above use the full
    # patient × timestep × feature SHAP values.
    # ========================================================

    reference_shap_feature = np.mean(

        np.abs(
            shap_reference
        ),

        axis=1
    )

    perturbed_shap_feature = np.mean(

        np.abs(
            shap_perturbed
        ),

        axis=1
    )

    reference_input_feature = np.mean(

        reference_sample,

        axis=1
    )

    perturbed_input_feature = np.mean(

        perturbed_sample,

        axis=1
    )

    # ========================================================
    # REFERENCE SUMMARY
    # ========================================================

    plt.figure()

    shap.summary_plot(

        reference_shap_feature,

        reference_input_feature,

        feature_names=feature_names,

        show=False
    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            out_dir,
            f"summary_reference_{suffix}.png"
        ),

        dpi=300,

        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # PERTURBED SUMMARY
    # ========================================================

    plt.figure()

    shap.summary_plot(

        perturbed_shap_feature,

        perturbed_input_feature,

        feature_names=feature_names,

        show=False
    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            out_dir,
            f"summary_perturbed_{suffix}.png"
        ),

        dpi=300,

        bbox_inches="tight"
    )

    plt.close()

    return {
        "comparison": comparison,
        "rho": rho,
        "p": p,
        "reference_df": reference_df,
        "perturbed_df": perturbed_df
    }


# ============================================================
# MAIN
# ============================================================

def main(
    return_df=False,
    cli_args=None
):

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--severity",

        type=str,

        default="Moderate",

        choices=[
            "Minimal",
            "Moderate",
            "Severe"
        ]
    )

    parser.add_argument(

        "--output_dir",

        type=str,

        default=RESULT_DIR
    )

    args = parser.parse_args(
        cli_args
    )

    severity = args.severity

    # ========================================================
    # READ TEST DATA
    # ========================================================

    test_reader = InHospitalMortalityReader(

        dataset_dir=os.path.join(
            DATA_DIR,
            "test"
        ),

        listfile=os.path.join(
            DATA_DIR,
            "test_listfile.csv"
        ),

        period_length=48.0
    )

    icu_map = build_icu_map(

        os.path.join(
            DATA_DIR,
            "../root"
        )
    )

    print(
        "Reading test data..."
    )

    test_ret = common_utils.read_chunk(

        test_reader,

        test_reader.get_number_of_examples()
    )

    X_raw = test_ret["X"]

    y = np.asarray(
        test_ret["y"]
    )

    names = test_ret["name"]

    feature_names = np.asarray(
        test_ret["header"]
    )

    icu = np.asarray([

        icu_map.get(
            n,
            "UNKNOWN"
        )

        for n in names

    ])

    # ========================================================
    # SICU MASK
    # ========================================================

    sicu_mask = np.array([

        str(x) == SICU_UNIT

        for x in icu

    ])

    print()
    print(
        "Total patients:",
        len(X_raw)
    )

    print(
        "SICU patients:",
        sicu_mask.sum()
    )

    print(
        "SICU fraction:",
        sicu_mask.mean()
    )

    if sicu_mask.sum() == 0:

        raise RuntimeError(
            "No SICU patients found."
        )

    # ========================================================
    # SICU BASELINE
    # ========================================================

    X_sicu_raw = [

        copy.deepcopy(
            X_raw[i]
        )

        for i in np.where(
            sicu_mask
        )[0]

    ]

    y_sicu = y[
        sicu_mask
    ]

    icu_sicu = icu[
        sicu_mask
    ]

    # ========================================================
    # SICU PERTURBED
    # ========================================================

    X_sicu_pert_raw = (
        apply_sicu_perturbation_raw(

            copy.deepcopy(
                X_sicu_raw
            ),

            icu_sicu,

            severity,

            feature_names
        )
    )

    # ========================================================
    # CLASSICAL FEATURE EXTRACTION
    # ========================================================

    X_sicu_base = (
        common_utils
        .extract_features_from_rawdata(

            X_sicu_raw,

            feature_names,

            "all",

            "all"
        )
    )

    X_sicu_pert = (
        common_utils
        .extract_features_from_rawdata(

            X_sicu_pert_raw,

            feature_names,

            "all",

            "all"
        )
    )

    # ========================================================
    # CLASSICAL MODEL PATHS
    # ========================================================

    classical_models = {

        "XGBoost":

            os.path.join(

                BASE_DIR,

                "mimic4models",

                "in_hospital_mortality",

                "XGBoostTuned",

                "xgb_pipeline.pkl"
            ),

        "LogisticRegression":

            os.path.join(

                BASE_DIR,

                "mimic4models",

                "in_hospital_mortality",

                "logistic",

                "logistic_pipeline.pkl"
            ),

        "Random Forest":

            os.path.join(

                BASE_DIR,

                "mimic4models",

                "in_hospital_mortality",

                "RandomForestTuned",

                "rf_pipeline.pkl"
            ),

        "MLP":

            os.path.join(

                BASE_DIR,

                "mimic4models",

                "in_hospital_mortality",

                "MLP",

                "mlp_pipeline.pkl"
            )
    }

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    severity_dir = os.path.join(

        args.output_dir,

        severity
    )

    os.makedirs(

        severity_dir,

        exist_ok=True
    )

    # ========================================================
    # CLASSICAL SHAP
    # ========================================================

    classical_results = {}

    for name, path in classical_models.items():

        print()
        print(
            "=========================================="
        )
        print(
            f"Running classical SHAP: {name}"
        )
        print(
            "=========================================="
        )

        model_dir = os.path.join(

            severity_dir,

            name.replace(
                " ",
                "_"
            )
        )

        os.makedirs(

            model_dir,

            exist_ok=True
        )

        result = run_classical_shap(

            model_name=name,

            pipeline_path=path,

            X_reference=X_sicu_base,

            X_perturbed=X_sicu_pert,

            icu_reference=icu_sicu,

            icu_perturbed=icu_sicu,

            out_dir=model_dir
        )

        classical_results[
            name
        ] = result

    # ========================================================
    # DEEP LEARNING SNAPSHOT
    # ========================================================

    snapshot_path = os.path.join(

        BASE_DIR,

        "mimic4models",

        "in_hospital_mortality",

        "LSTM",

        "test_snapshot.pkl"
    )

    snapshot = joblib.load(
        snapshot_path
    )

    X_norm = np.asarray(
        snapshot["X"],
        dtype=np.float32
    )

    snapshot_icu = np.asarray(
        snapshot["icu"]
    )

    # ========================================================
    # VALIDATE SNAPSHOT
    # ========================================================

    if len(snapshot_icu) != len(X_norm):

        raise RuntimeError(
            "Snapshot ICU length does not "
            "match snapshot X."
        )

    # ========================================================
    # LSTM PIPELINE
    # ========================================================

    lstm_pipeline_path = os.path.join(

        BASE_DIR,

        "mimic4models",

        "in_hospital_mortality",

        "LSTM_DS",

        "preprocessing_pipeline.pkl"
    )

    lstm_pipeline = joblib.load(
        lstm_pipeline_path
    )

    deep_feature_names = np.asarray(

        lstm_pipeline[
            "feature_names"
        ]
    )

    if X_norm.shape[2] != len(
        deep_feature_names
    ):

        raise RuntimeError(

            "Deep snapshot feature count "
            "does not match LSTM feature names: "

            f"{X_norm.shape[2]} vs "
            f"{len(deep_feature_names)}"
        )

    # ========================================================
    # NORMALIZER
    # ========================================================

    if "normalizer" in lstm_pipeline:

        normalizer = (
            lstm_pipeline["normalizer"]
        )

    elif "normalizer_path" in lstm_pipeline:

        normalizer = joblib.load(

            lstm_pipeline[
                "normalizer_path"
            ]
        )

    else:

        normalizer_path = os.path.join(

            BASE_DIR,

            "mimic4models",

            "in_hospital_mortality",

            "LSTM_DS",

            "normalizer.pkl"
        )

        if not os.path.exists(
            normalizer_path
        ):

            raise RuntimeError(

                "Could not find normalizer in "
                "LSTM_DS preprocessing pipeline "
                "or at the expected normalizer.pkl path."
            )

        normalizer = joblib.load(
            normalizer_path
        )

    # ========================================================
    # CONTINUOUS CHANNELS
    # ========================================================

    if "cont_channels" in lstm_pipeline:

        cont_channels = (
            lstm_pipeline[
                "cont_channels"
            ]
        )

    else:

        cont_channels = np.arange(
            X_norm.shape[2]
        )

    cont_channels = np.asarray(
        cont_channels,
        dtype=int
    )

    # ========================================================
    # SELECT SICU PATIENTS
    # ========================================================

    snapshot_sicu_mask = np.array([

        str(x) == SICU_UNIT

        for x in snapshot_icu

    ])

    X_deep_sicu = X_norm[
        snapshot_sicu_mask
    ]

    icu_deep_sicu = snapshot_icu[
        snapshot_sicu_mask
    ]

    print()
    print(
        "Deep-learning SICU patients:",
        len(X_deep_sicu)
    )

    if len(X_deep_sicu) == 0:

        raise RuntimeError(
            "No SICU patients found "
            "in deep-learning snapshot."
        )

    # ========================================================
    # CREATE PERTURBED DEEP TENSOR
    # ========================================================

    np.random.seed(
        RANDOM_SEED
    )

    X_deep_sicu_pert = (
        apply_sicu_perturbation_tensor(

            X=X_deep_sicu,

            icu=icu_deep_sicu,

            severity=severity,

            feature_names=deep_feature_names,

            normalizer=normalizer,

            cont_channels=cont_channels
        )
    )

    # ========================================================
    # VERIFY PERTURBATION
    # ========================================================

    if X_deep_sicu_pert.shape != X_deep_sicu.shape:

        raise RuntimeError(

            "Perturbed deep tensor shape changed: "

            f"{X_deep_sicu.shape} -> "
            f"{X_deep_sicu_pert.shape}"
        )

    if np.isnan(
        X_deep_sicu_pert
    ).any():

        raise RuntimeError(

            "NaNs introduced during "
            "SICU deep perturbation."
        )

    # ========================================================
    # DEEP MODEL CONFIGURATION
    #
    # IMPORTANT:
    #
    # LSTM_DS is kept at the exact filename you supplied.
    #
    # The other three models are discovered from their
    # respective keras_states directories.
    # ========================================================

    deep_model_paths = {

        "LSTM":

            None,

        "LSTM_DS":

            os.path.join(

                BASE_DIR,

                "mimic4models",

                "in_hospital_mortality",

                "LSTM_DS",

                "keras_states",

                "k_lstm.n16.d0.3.dep2.bs8.ts1.0."
                "trc0.5.epoch69.test0.2358170449733734.keras"
            ),

        "ChannelwiseLSTM":

            None,

        "ChannelwiseLSTM_DS":

            None
    }

    # ========================================================
    # DISCOVER MISSING WEIGHT PATHS
    # ========================================================

    for model_name in [

        "LSTM",
        "ChannelwiseLSTM",
        "ChannelwiseLSTM_DS"

    ]:

        deep_model_paths[
            model_name
        ] = find_deep_model_weights(
            model_name
        )

    # ========================================================
    # DEEP PIPELINES
    #
    # If separate preprocessing pipelines exist for the deep
    # models, use them. Otherwise the LSTM_DS pipeline is used,
    # because all four models operate on the same 48 x 96
    # normalized tensor representation.
    # ========================================================

    deep_pipeline_paths = {

        "LSTM":
            lstm_pipeline_path,

        "LSTM_DS":
            lstm_pipeline_path,

        "ChannelwiseLSTM":
            lstm_pipeline_path,

        "ChannelwiseLSTM_DS":
            lstm_pipeline_path
    }

    # ========================================================
    # RUN ALL FOUR DEEP MODELS
    # ========================================================

    deep_results = {}

    for model_name in [

        "LSTM",
        "LSTM_DS",
        "ChannelwiseLSTM",
        "ChannelwiseLSTM_DS"

    ]:

        model_dir = os.path.join(

            severity_dir,

            model_name
        )

        os.makedirs(

            model_dir,

            exist_ok=True
        )

        result = run_deep_shap(

            model_name=model_name,

            model_path=deep_model_paths[
                model_name
            ],

            pipeline_path=deep_pipeline_paths[
                model_name
            ],

            X_reference=X_deep_sicu,

            X_perturbed=X_deep_sicu_pert,

            out_dir=model_dir,

            severity=severity
        )

        deep_results[
            model_name
        ] = result

    # ========================================================
    # ALL 8 MODELS
    # ========================================================

    all_results = {}

    all_results.update(
        classical_results
    )

    all_results.update(
        deep_results
    )

    # ========================================================
    # FINAL SUMMARY
    #
    # EXACTLY ONE ROW PER MODEL.
    #
    # Computed directly from the in-memory reference/perturbed
    # feature-importance tables returned by compare_shap, the
    # same way MICU_SHAP.py builds its summary — rather than
    # re-reading CSVs back off disk. This guarantees the
    # Top10_Overlap figure here matches whatever was printed
    # by print_shap_results() for that model.
    # ========================================================

    model_names = [

        "XGBoost",

        "LogisticRegression",

        "Random Forest",

        "MLP",

        "LSTM",

        "LSTM_DS",

        "ChannelwiseLSTM",

        "ChannelwiseLSTM_DS"
    ]

    summary_rows = []

    for model_name in model_names:

        result = all_results.get(
            model_name
        )

        if result is None:

            print(
                f"WARNING: no SHAP result found "
                f"for {model_name}"
            )

            continue

        overlap_count = len(
            set(
                result[
                    "reference_df"
                ].head(10)["Feature"]
            )
            &
            set(
                result[
                    "perturbed_df"
                ].head(10)["Feature"]
            )
        )

        summary_rows.append({

            "Model":
                model_name,

            "Severity":
                severity,

            "Spearman_Rho":
                result["rho"],

            "P_Value":
                result["p"],

            "Top10_Overlap":
                overlap_count,

            "Top10_Overlap_Fraction":
                overlap_count / 10

        })

    # ========================================================
    # SUMMARY DATAFRAME
    # ========================================================

    summary = pd.DataFrame(
        summary_rows
    )

    # ========================================================
    # ENFORCE MODEL ORDER
    # ========================================================

    summary["Model"] = pd.Categorical(

        summary["Model"],

        categories=model_names,

        ordered=True
    )

    summary = summary.sort_values(
        "Model"
    ).reset_index(
        drop=True
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    summary_csv_path = os.path.join(

        severity_dir,

        "sicu_shap_summary.csv"
    )

    summary.to_csv(

        summary_csv_path,

        index=False
    )

    # ========================================================
    # SAVE LATEX
    # ========================================================

    latex_table = summary.to_latex(

        index=False,

        escape=True,

        float_format=lambda x: (

            f"{x:.6f}"

            if isinstance(
                x,
                (float, np.floating)
            )
            and
            np.isfinite(x)

            else str(x)

        )
    )

    latex_path = os.path.join(

        severity_dir,

        "sicu_shap_summary.tex"
    )

    with open(

        latex_path,

        "w",

        encoding="utf-8"

    ) as f:

        f.write(
            latex_table
        )

    # ========================================================
    # ALSO SAVE A COMPACT PAPER TABLE
    #
    # This is useful directly for Chapter 5/Results.
    # ========================================================

    paper_summary = summary.copy()

    paper_summary[
        "Spearman_Rho"
    ] = paper_summary[
        "Spearman_Rho"
    ].round(3)

    paper_summary[
        "Top10_Overlap"
    ] = paper_summary[
        "Top10_Overlap"
    ].astype("Int64")

    paper_summary[
        "Top10_Overlap_Fraction"
    ] = paper_summary[
        "Top10_Overlap_Fraction"
    ].round(2)

    paper_summary_path = os.path.join(

        severity_dir,

        "sicu_shap_summary_compact.csv"
    )

    paper_summary.to_csv(

        paper_summary_path,

        index=False
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print(
        "======================================"
    )

    print(
        f"SICU SHAP RESULTS — {severity}"
    )

    print(
        "======================================"
    )

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Saved summary:"
    )

    print(
        summary_csv_path
    )

    print(
        latex_path
    )

    print()

    # ========================================================
    # RETURN
    # ========================================================

    if return_df:

        return summary


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
import os
import sys
import copy
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
    "micu_sicu_shap"
)

sys.path.append(
    os.path.join(
        BASE_DIR,
        "mimic4models"
    )
)


# ============================================================
# MICU/SICU UNIT
# ============================================================

MICU_SICU_UNIT = (
    "Medical/Surgical Intensive Care Unit (MICU/SICU)"
)


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


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
# MICU/SICU PERTURBATION PARAMETERS
# ============================================================

MICU_SICU_PARAMS = {

    "Minimal": {

        "creatinine_pct": (0.10, 0.20),
        "potassium_pct": (0.00, 0.05),
        "sodium_pct": (-0.02, 0.02),

        "hr_shift": (3, 5),
        "rr_shift": (1, 3),

        "mbp_pct": (-0.05, -0.02),
        "sbp_pct": None,

        "lactate_pct": None,
        "spo2_pct": None,
    },

    "Moderate": {

        "creatinine_pct": (0.30, 0.60),
        "potassium_pct": (0.10, 0.25),
        "sodium_pct": (-0.05, -0.02),

        "hr_shift": (5, 15),
        "rr_shift": (3, 6),

        "mbp_pct": (-0.10, -0.05),
        "sbp_pct": (-0.10, -0.05),

        "lactate_pct": (0.10, 0.25),
        "spo2_pct": None,
    },

    "Severe": {

        "creatinine_pct": (0.80, 1.50),
        "potassium_pct": (0.25, 0.60),
        "sodium_pct": (-0.10, -0.05),

        "hr_shift": (10, 25),
        "rr_shift": (5, 10),

        "mbp_pct": (-0.20, -0.10),
        "sbp_pct": (-0.25, -0.10),

        "lactate_pct": (0.25, 0.75),
        "spo2_pct": (-0.08, -0.02),
    }
}


# ============================================================
# RAW MICU/SICU PERTURBATION
# ============================================================

def apply_micu_sicu_perturbation_raw(
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

    params = MICU_SICU_PARAMS[
        severity
    ]

    # --------------------------------------------------------
    # Required features
    # --------------------------------------------------------

    hr_i = find_feature(
        feature_names,
        ["Heart Rate"]
    )

    rr_i = find_feature(
        feature_names,
        ["Respiratory rate"]
    )

    # --------------------------------------------------------
    # Optional features
    # --------------------------------------------------------

    def try_find(names):

        try:

            return (
                find_feature(
                    feature_names,
                    names
                ),
                True
            )

        except ValueError:

            return (
                None,
                False
            )

    cre_i, has_cre = try_find(
        ["Creatinine"]
    )

    pot_i, has_pot = try_find(
        ["Potassium"]
    )

    sod_i, has_sod = try_find(
        ["Sodium"]
    )

    mbp_i, has_mbp = try_find(
        ["Mean blood pressure"]
    )

    sbp_i, has_sbp = try_find(
        ["Systolic blood pressure"]
    )

    lac_i, has_lac = try_find(
        ["Lactate"]
    )

    spo2_i, has_spo2 = try_find(
        ["Oxygen saturation"]
    )

    # --------------------------------------------------------
    # Patient-level perturbation
    # --------------------------------------------------------

    for patient, unit in zip(
        X,
        icu
    ):

        if str(unit) != MICU_SICU_UNIT:
            continue

        # One random perturbation magnitude per patient
        cre_pct = (
            np.random.uniform(
                *params["creatinine_pct"]
            )
            if has_cre
            else 0
        )

        pot_pct = (
            np.random.uniform(
                *params["potassium_pct"]
            )
            if has_pot
            else 0
        )

        sod_pct = (
            np.random.uniform(
                *params["sodium_pct"]
            )
            if has_sod
            else 0
        )

        hr_add = np.random.uniform(
            *params["hr_shift"]
        )

        rr_add = np.random.uniform(
            *params["rr_shift"]
        )

        mbp_pct = (
            np.random.uniform(
                *params["mbp_pct"]
            )
            if has_mbp
            else 0
        )

        sbp_pct = (
            np.random.uniform(
                *params["sbp_pct"]
            )
            if (
                has_sbp
                and
                params["sbp_pct"] is not None
            )
            else 0
        )

        lac_pct = (
            np.random.uniform(
                *params["lactate_pct"]
            )
            if (
                has_lac
                and
                params["lactate_pct"] is not None
            )
            else 0
        )

        spo2_pct = (
            np.random.uniform(
                *params["spo2_pct"]
            )
            if (
                has_spo2
                and
                params["spo2_pct"] is not None
            )
            else 0
        )

        # ----------------------------------------------------
        # Apply perturbations
        # ----------------------------------------------------

        for row in patient:

            # Creatinine
            if (
                has_cre
                and
                row[cre_i] != ""
            ):

                v = safe_float(
                    row[cre_i]
                )

                if (
                    np.isfinite(v)
                    and
                    0.1 <= v <= 30
                ):

                    row[cre_i] = str(
                        np.clip(
                            v * (1 + cre_pct),
                            0.1,
                            30
                        )
                    )

            # Potassium
            if (
                has_pot
                and
                row[pot_i] != ""
            ):

                v = safe_float(
                    row[pot_i]
                )

                if (
                    np.isfinite(v)
                    and
                    1.0 <= v <= 10.0
                ):

                    row[pot_i] = str(
                        np.clip(
                            v * (1 + pot_pct),
                            1.0,
                            10.0
                        )
                    )

            # Sodium
            if (
                has_sod
                and
                row[sod_i] != ""
            ):

                v = safe_float(
                    row[sod_i]
                )

                if (
                    np.isfinite(v)
                    and
                    110 <= v <= 170
                ):

                    row[sod_i] = str(
                        np.clip(
                            v * (1 + sod_pct),
                            110,
                            170
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
                            v + hr_add,
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
                    2 <= v <= 80
                ):

                    row[rr_i] = str(
                        np.clip(
                            v + rr_add,
                            2,
                            80
                        )
                    )

            # Mean blood pressure
            if (
                has_mbp
                and
                row[mbp_i] != ""
            ):

                v = safe_float(
                    row[mbp_i]
                )

                if (
                    np.isfinite(v)
                    and
                    20 <= v <= 200
                ):

                    row[mbp_i] = str(
                        np.clip(
                            v * (1 + mbp_pct),
                            20,
                            200
                        )
                    )

            # Systolic BP
            if (
                has_sbp
                and
                params["sbp_pct"] is not None
                and
                row[sbp_i] != ""
            ):

                v = safe_float(
                    row[sbp_i]
                )

                if (
                    np.isfinite(v)
                    and
                    40 <= v <= 300
                ):

                    row[sbp_i] = str(
                        np.clip(
                            v * (1 + sbp_pct),
                            40,
                            300
                        )
                    )

            # Lactate
            if (
                has_lac
                and
                params["lactate_pct"] is not None
                and
                row[lac_i] != ""
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
                            v * (1 + lac_pct),
                            0.1,
                            30
                        )
                    )

            # SpO2
            if (
                has_spo2
                and
                params["spo2_pct"] is not None
                and
                row[spo2_i] != ""
            ):

                v = safe_float(
                    row[spo2_i]
                )

                if (
                    np.isfinite(v)
                    and
                    50 <= v <= 100
                ):

                    row[spo2_i] = str(
                        np.clip(
                            v * (1 + spo2_pct),
                            50,
                            100
                        )
                    )

    return X


# ============================================================
# NORMALIZATION
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
# TENSOR MICU/SICU PERTURBATION
#
# Input:
#     normalized tensor
#
# Steps:
#     normalized
#         ↓
#     clinical units
#         ↓
#     perturb
#         ↓
#     normalized
#
# Missing-value zeros are preserved.
# ============================================================

def apply_micu_sicu_perturbation_tensor(
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

    micu_sicu_mask = np.array([

        str(x) == MICU_SICU_UNIT

        for x in icu

    ])

    if micu_sicu_mask.sum() == 0:

        return X

    # --------------------------------------------------------
    # Convert normalized data to clinical units
    # --------------------------------------------------------

    X_raw = inverse_normalize(

        X,

        normalizer,

        cont_channels
    )

    params = MICU_SICU_PARAMS[
        severity
    ]

    # --------------------------------------------------------
    # Feature indices
    # --------------------------------------------------------

    hr_i = find_feature(
        feature_names,
        ["Heart Rate"]
    )

    rr_i = find_feature(
        feature_names,
        ["Respiratory rate"]
    )

    def try_find(names):

        try:

            return (
                find_feature(
                    feature_names,
                    names
                ),
                True
            )

        except ValueError:

            return (
                None,
                False
            )

    cre_i, has_cre = try_find(
        ["Creatinine"]
    )

    pot_i, has_pot = try_find(
        ["Potassium"]
    )

    sod_i, has_sod = try_find(
        ["Sodium"]
    )

    mbp_i, has_mbp = try_find(
        ["Mean blood pressure"]
    )

    sbp_i, has_sbp = try_find(
        ["Systolic blood pressure"]
    )

    lac_i, has_lac = try_find(
        ["Lactate"]
    )

    spo2_i, has_spo2 = try_find(
        ["Oxygen saturation"]
    )

    # --------------------------------------------------------
    # Patient-level perturbation values
    # --------------------------------------------------------

    n_patients = micu_sicu_mask.sum()

    def draw_pct(bounds):

        return np.random.uniform(
            *bounds,
            size=n_patients
        )[:, None]

    def draw_abs(bounds):

        return np.random.uniform(
            *bounds,
            size=n_patients
        )[:, None]

    # --------------------------------------------------------
    # Percentage application
    # --------------------------------------------------------

    def pct_apply(
        feature_i,
        pct_arr,
        low,
        high
    ):

        arr = X_raw[
            micu_sicu_mask,
            :,
            feature_i
        ]

        valid = arr != 0

        arr = np.where(

            valid,

            np.clip(
                arr * (1 + pct_arr),
                low,
                high
            ),

            arr
        )

        X_raw[
            micu_sicu_mask,
            :,
            feature_i
        ] = arr

    # --------------------------------------------------------
    # Absolute application
    # --------------------------------------------------------

    def abs_apply(
        feature_i,
        add_arr,
        low,
        high
    ):

        arr = X_raw[
            micu_sicu_mask,
            :,
            feature_i
        ]

        valid = arr != 0

        arr = np.where(

            valid,

            np.clip(
                arr + add_arr,
                low,
                high
            ),

            arr
        )

        X_raw[
            micu_sicu_mask,
            :,
            feature_i
        ] = arr

    # --------------------------------------------------------
    # Creatinine
    # --------------------------------------------------------

    if has_cre:

        pct_apply(

            cre_i,

            draw_pct(
                params["creatinine_pct"]
            ),

            0.1,
            30
        )

    # --------------------------------------------------------
    # Potassium
    # --------------------------------------------------------

    if has_pot:

        pct_apply(

            pot_i,

            draw_pct(
                params["potassium_pct"]
            ),

            1.0,
            10.0
        )

    # --------------------------------------------------------
    # Sodium
    # --------------------------------------------------------

    if has_sod:

        pct_apply(

            sod_i,

            draw_pct(
                params["sodium_pct"]
            ),

            110,
            170
        )

    # --------------------------------------------------------
    # Heart rate
    # --------------------------------------------------------

    abs_apply(

        hr_i,

        draw_abs(
            params["hr_shift"]
        ),

        20,
        250
    )

    # --------------------------------------------------------
    # Respiratory rate
    # --------------------------------------------------------

    abs_apply(

        rr_i,

        draw_abs(
            params["rr_shift"]
        ),

        2,
        80
    )

    # --------------------------------------------------------
    # Mean BP
    # --------------------------------------------------------

    if has_mbp:

        pct_apply(

            mbp_i,

            draw_pct(
                params["mbp_pct"]
            ),

            20,
            200
        )

    # --------------------------------------------------------
    # Systolic BP
    # --------------------------------------------------------

    if (
        has_sbp
        and
        params["sbp_pct"] is not None
    ):

        pct_apply(

            sbp_i,

            draw_pct(
                params["sbp_pct"]
            ),

            40,
            300
        )

    # --------------------------------------------------------
    # Lactate
    # --------------------------------------------------------

    if (
        has_lac
        and
        params["lactate_pct"] is not None
    ):

        pct_apply(

            lac_i,

            draw_pct(
                params["lactate_pct"]
            ),

            0.1,
            30
        )

    # --------------------------------------------------------
    # SpO2
    # --------------------------------------------------------

    if (
        has_spo2
        and
        params["spo2_pct"] is not None
    ):

        pct_apply(

            spo2_i,

            draw_pct(
                params["spo2_pct"]
            ),

            50,
            100
        )

    # --------------------------------------------------------
    # Re-normalize
    # --------------------------------------------------------

    X_perturbed = renormalize(

        X_raw,

        normalizer,

        cont_channels
    )

    return X_perturbed


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
        pipeline[
            "correlated_columns_removed"
        ]
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
        ).reshape(
            -1,
            1
        )
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
        pipeline[
            "correlated_columns_removed"
        ]
    ) > 0:

        drop = set(
            pipeline[
                "correlated_columns_removed"
            ]
        )

        feature_names = [

            f

            for f in feature_names

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
# MODEL PROBABILITY
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
# SHAP NORMALIZATION
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
            "shape (N,T,F). Got "
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

    # --------------------------------------------------------
    # Save feature importance
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Spearman rank correlation
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Rank scatter
    # --------------------------------------------------------

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
        f"MICU/SICU SHAP Rank Stability\n"
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
            len(overlap)
        ],

        "Overlap_Fraction": [
            len(overlap) / 10
        ]

    }).to_csv(

        os.path.join(
            out_dir,
            f"top10_overlap_{suffix}.csv"
        ),

        index=False
    )

    # --------------------------------------------------------
    # SHAP change plot
    # --------------------------------------------------------

    plot_df = comparison.sort_values(

        "SHAP_Change",
        ascending=False

    ).head(
        min(15, len(comparison))
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
        "MICU/SICU Feature Attribution Change\n"
        "Baseline → MICU/SICU AKI Perturbation"
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

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

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

    feature_names = (
        get_classical_feature_names(
            pipeline
        )
    )

    if len(feature_names) != X_ref.shape[1]:

        raise RuntimeError(

            f"{model_name}: feature-name count "
            f"{len(feature_names)} does not match "
            f"transformed feature count "
            f"{X_ref.shape[1]}"
        )

    explain_model = (
        unwrap_calibrated_model(
            model
        )
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Save raw SHAP
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

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
# LOAD LSTM MODULE
# ============================================================

def load_lstm_module():

    module_path = os.path.join(

        BASE_DIR,

        "mimic4models",

        "keras_models",

        "lstm.py"
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            "mimic_lstm",
            module_path
        )
    )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# LOAD CHANNELWISE LSTM MODULE
# ============================================================

def load_channelwise_module():

    module_path = os.path.join(

        BASE_DIR,

        "mimic4models",

        "keras_models",

        "channel_wise_lstms.py"
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            "mimic_channelwise",
            module_path
        )
    )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# BUILD DEEP MODEL
# ============================================================

def build_deep_model(
    model_name,
    pipeline
):

    if model_name in [
        "LSTM",
        "LSTM_DS"
    ]:

        lstm_module = (
            load_lstm_module()
        )

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

            input_dim=96
        )

        return model

    if model_name in [
        "ChannelwiseLSTM",
        "ChannelwiseLSTM_DS"
    ]:

        channelwise_module = (
            load_channelwise_module()
        )

        target_repl = (
            model_name == "ChannelwiseLSTM_DS"
        )

        model = channelwise_module.Network(

            dim=16,

            batch_norm=False,

            dropout=0.3,

            rec_dropout=0.0,

            header=pipeline[
                "feature_names"
            ],

            task="ihm",

            target_repl=target_repl,

            deep_supervision=False,

            depth=2,

            input_dim=96,

            size_coef=4.0
        )

        return model

    raise ValueError(
        f"Unsupported deep model: "
        f"{model_name}"
    )


# ============================================================
# DEEP SHAP
#
# Actual 48 × 96 trajectories.
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

    # --------------------------------------------------------
    # Load pipeline
    # --------------------------------------------------------

    pipeline = joblib.load(
        pipeline_path
    )

    feature_names = list(
        pipeline[
            "feature_names"
        ]
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_deep_model(

        model_name,

        pipeline
    )

    model.load_weights(
        model_path
    )

    # --------------------------------------------------------
    # Convert arrays
    # --------------------------------------------------------

    X_reference = np.asarray(

        X_reference,

        dtype=np.float32
    )

    X_perturbed = np.asarray(

        X_perturbed,

        dtype=np.float32
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

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

    n_timesteps = (
        X_reference.shape[1]
    )

    n_features = (
        X_reference.shape[2]
    )

    if len(feature_names) != n_features:

        raise ValueError(

            f"Number of feature names "
            f"({len(feature_names)}) does not "
            f"match input features "
            f"({n_features})."
        )

    print()
    print(
        f"{model_name} deep SHAP input shape:",
        X_reference.shape
    )

    # --------------------------------------------------------
    # Paired sampling
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Fixed-shape Keras input
    # --------------------------------------------------------

    fixed_input = tf.keras.Input(

        shape=(

            n_timesteps,

            n_features
        ),

        dtype=tf.float32,

        name=f"{model_name}_shap_input"
    )

    outputs = model(
        fixed_input,
        training=False
    )

    # --------------------------------------------------------
    # Deep models can return multiple outputs
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Fixed model
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Prediction sanity check
    # --------------------------------------------------------

    test_prediction = (
        fixed_model.predict(
            background[:2],
            verbose=0
        )
    )

    if test_prediction.shape != (
        2,
        1
    ):

        raise ValueError(

            "Wrapped model does not produce "
            "expected (N,1) output. Got "
            f"{test_prediction.shape}"
        )

    # --------------------------------------------------------
    # Gradient SHAP
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Reference SHAP
    # --------------------------------------------------------

    print(
        "\nCalculating reference SHAP..."
    )

    shap_reference_raw = (
        explainer.shap_values(

            reference_sample,

            nsamples=200
        )
    )

    # --------------------------------------------------------
    # Perturbed SHAP
    # --------------------------------------------------------

    print(
        "Calculating perturbed SHAP..."
    )

    shap_perturbed_raw = (
        explainer.shap_values(

            perturbed_sample,

            nsamples=200
        )
    )

    # --------------------------------------------------------
    # Normalize SHAP output
    # --------------------------------------------------------

    def process_deep_shap(
        values
    ):

        if isinstance(
            values,
            list
        ):

            if len(values) != 1:

                raise ValueError(

                    "Expected exactly one "
                    "SHAP output."
                )

            values = values[0]

        values = np.asarray(
            values
        )

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

    shap_reference = (
        process_deep_shap(
            shap_reference_raw
        )
    )

    shap_perturbed = (
        process_deep_shap(
            shap_perturbed_raw
        )
    )

    print(
        "Reference SHAP shape:",
        shap_reference.shape
    )

    print(
        "Perturbed SHAP shape:",
        shap_perturbed.shape
    )

    # --------------------------------------------------------
    # Save raw temporal SHAP
    # --------------------------------------------------------

    np.save(

        os.path.join(
            out_dir,
            f"shap_reference_"
            f"{model_name.lower()}_"
            f"{severity.lower()}.npy"
        ),

        shap_reference
    )

    np.save(

        os.path.join(
            out_dir,
            f"shap_perturbed_"
            f"{model_name.lower()}_"
            f"{severity.lower()}.npy"
        ),

        shap_perturbed
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    suffix = (

        f"{model_name.lower()}_"
        f"{severity.lower()}"
    )

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

    # --------------------------------------------------------
    # Temporal aggregation ONLY for summary plots
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Reference summary
    # --------------------------------------------------------

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

            f"summary_reference_"
            f"{model_name.lower()}_"
            f"{severity.lower()}.png"
        ),

        dpi=300,

        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Perturbed summary
    # --------------------------------------------------------

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

            f"summary_perturbed_"
            f"{model_name.lower()}_"
            f"{severity.lower()}.png"
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

    X_raw = test_ret[
        "X"
    ]

    y = np.asarray(
        test_ret[
            "y"
        ]
    )

    names = test_ret[
        "name"
    ]

    feature_names = np.asarray(
        test_ret[
            "header"
        ]
    )

    icu = np.asarray([

        icu_map.get(
            n,
            "UNKNOWN"
        )

        for n in names

    ])

    # ========================================================
    # MICU/SICU MASK
    # ========================================================

    micu_sicu_mask = np.array([

        str(x) == MICU_SICU_UNIT

        for x in icu

    ])

    print()
    print(
        "Total patients:",
        len(X_raw)
    )

    print(
        "MICU/SICU patients:",
        micu_sicu_mask.sum()
    )

    print(
        "MICU/SICU fraction:",
        micu_sicu_mask.mean()
    )

    if micu_sicu_mask.sum() == 0:

        raise RuntimeError(
            "No MICU/SICU patients found."
        )

    # ========================================================
    # MICU/SICU BASELINE
    # ========================================================

    X_micu_sicu_raw = [

        copy.deepcopy(
            X_raw[i]
        )

        for i in np.where(
            micu_sicu_mask
        )[0]

    ]

    y_micu_sicu = y[
        micu_sicu_mask
    ]

    icu_micu_sicu = icu[
        micu_sicu_mask
    ]

    # ========================================================
    # MICU/SICU PERTURBED
    # ========================================================

    np.random.seed(
        RANDOM_SEED
    )

    X_micu_sicu_pert_raw = (
        apply_micu_sicu_perturbation_raw(

            copy.deepcopy(
                X_micu_sicu_raw
            ),

            icu_micu_sicu,

            severity,

            feature_names
        )
    )

    # ========================================================
    # CLASSICAL FEATURE EXTRACTION
    # ========================================================

    X_micu_sicu_base = (

        common_utils
        .extract_features_from_rawdata(

            X_micu_sicu_raw,

            feature_names,

            "all",

            "all"
        )
    )

    X_micu_sicu_pert = (

        common_utils
        .extract_features_from_rawdata(

            X_micu_sicu_pert_raw,

            feature_names,

            "all",

            "all"
        )
    )

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
    # CLASSICAL SHAP
    # ========================================================

    results = {}

    for model_name, pipeline_path in (
        classical_models.items()
    ):

        print()
        print(
            f"Running classical SHAP: "
            f"{model_name}"
        )

        model_dir = os.path.join(

            severity_dir,

            model_name.replace(
                " ",
                "_"
            )
        )

        os.makedirs(

            model_dir,

            exist_ok=True
        )

        results[model_name] = (
            run_classical_shap(

                model_name=model_name,

                pipeline_path=pipeline_path,

                X_reference=X_micu_sicu_base,

                X_perturbed=X_micu_sicu_pert,

                icu_reference=icu_micu_sicu,

                icu_perturbed=icu_micu_sicu,

                out_dir=model_dir
            )
        )

    # ========================================================
    # DEEP SNAPSHOT
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
    # DEEP MODEL PATHS
    # ========================================================

    deep_models = {

        "LSTM": os.path.join(

            BASE_DIR,

            "mimic4models",

            "in_hospital_mortality",

            "keras_states",

            "k_lstm.n16.d0.3.dep2.bs8.ts1.0."
            "epoch95.test0.23363609611988068.keras"
        ),

        "LSTM_DS": os.path.join(

            BASE_DIR,

            "mimic4models",

            "in_hospital_mortality",

            "LSTM_DS",

            "keras_states",

            "k_lstm.n16.d0.3.dep2.bs8.ts1.0."
            "trc0.5.epoch69.test0.2358170449733734.keras"
        ),

        "ChannelwiseLSTM": os.path.join(

            BASE_DIR,

            "mimic4models",

            "in_hospital_mortality",

            "ChannelwiseLSTM",

            "keras_states",

            "k_channel_wise_lstms.n16."
            "szc4.0.d0.3.dep2.bs8.ts1.0."
            "epoch13.test0.2362385094165802.keras"
        ),

        "ChannelwiseLSTM_DS": os.path.join(

            BASE_DIR,

            "mimic4models",

            "in_hospital_mortality",

            "ChannelwiseLSTM_DS",

            "keras_states",

            "k_channel_wise_lstms.n16."
            "szc4.0.d0.3.dep2.bs8.ts1.0."
            "trc0.5.epoch59."
            "test0.2366071343421936.keras"
        )
    }

    # ========================================================
    # DEEP PIPELINES
    # ========================================================

    deep_pipelines = {

        "LSTM": os.path.join(

            BASE_DIR,

            "mimic4models",

            "in_hospital_mortality",

            "LSTM",

            "preprocessing_pipeline.pkl"
        ),

        "LSTM_DS": os.path.join(

            BASE_DIR,

            "mimic4models",

            "in_hospital_mortality",

            "LSTM_DS",

            "preprocessing_pipeline.pkl"
        ),

        "ChannelwiseLSTM": os.path.join(

            BASE_DIR,

            "mimic4models",

            "in_hospital_mortality",

            "ChannelwiseLSTM",

            "preprocessing_pipeline.pkl"
        ),

        "ChannelwiseLSTM_DS": os.path.join(

            BASE_DIR,

            "mimic4models",

            "in_hospital_mortality",

            "ChannelwiseLSTM_DS",

            "preprocessing_pipeline.pkl"
        )
    }

    # ========================================================
    # RUN DEEP SHAP
    # ========================================================

    snapshot_micu_sicu_mask = np.array([

        str(x) == MICU_SICU_UNIT

        for x in snapshot_icu

    ])

    X_deep_micu_sicu = X_norm[
        snapshot_micu_sicu_mask
    ]

    icu_deep_micu_sicu = snapshot_icu[
        snapshot_micu_sicu_mask
    ]

    print()
    print(
        "Deep-learning MICU/SICU patients:",
        len(X_deep_micu_sicu)
    )

    if len(X_deep_micu_sicu) == 0:

        raise RuntimeError(

            "No MICU/SICU patients found "
            "in deep-learning snapshot."
        )

    for model_name in deep_models:

        print()
        print(
            f"Running deep SHAP: "
            f"{model_name}"
        )

        pipeline = joblib.load(

            deep_pipelines[
                model_name
            ]
        )

        deep_feature_names = np.asarray(

            pipeline[
                "feature_names"
            ]
        )

        normalizer = pipeline[
            "normalizer"
        ]

        cont_channels = np.asarray(

            pipeline[
                "cont_channels"
            ],

            dtype=int
        )

        if X_deep_micu_sicu.shape[2] != len(
            deep_feature_names
        ):

            raise RuntimeError(

                f"{model_name}: snapshot feature "
                f"count {X_deep_micu_sicu.shape[2]} "
                f"does not match pipeline feature "
                f"count {len(deep_feature_names)}"
            )

        # ----------------------------------------------------
        # Create perturbation in clinical space
        # ----------------------------------------------------

        np.random.seed(
            RANDOM_SEED
        )

        X_deep_micu_sicu_pert = (
            apply_micu_sicu_perturbation_tensor(

                X=X_deep_micu_sicu,

                icu=icu_deep_micu_sicu,

                severity=severity,

                feature_names=deep_feature_names,

                normalizer=normalizer,

                cont_channels=cont_channels
            )
        )

        if (
            X_deep_micu_sicu_pert.shape
            !=
            X_deep_micu_sicu.shape
        ):

            raise RuntimeError(

                f"{model_name}: perturbed tensor "
                f"shape changed: "
                f"{X_deep_micu_sicu.shape} -> "
                f"{X_deep_micu_sicu_pert.shape}"
            )

        if np.isnan(
            X_deep_micu_sicu_pert
        ).any():

            raise RuntimeError(

                f"{model_name}: NaNs introduced "
                "during perturbation."
            )

        model_dir = os.path.join(

            severity_dir,

            model_name
        )

        os.makedirs(

            model_dir,

            exist_ok=True
        )

        results[model_name] = (
            run_deep_shap(

                model_name=model_name,

                model_path=deep_models[
                    model_name
                ],

                pipeline_path=deep_pipelines[
                    model_name
                ],

                X_reference=X_deep_micu_sicu,

                X_perturbed=X_deep_micu_sicu_pert,

                out_dir=model_dir,

                severity=severity
            )
        )

    # ========================================================
    # COMBINED SUMMARY
    # ========================================================

    summary_rows = []

    for model_name, result in (
        results.items()
    ):

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
                len(
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
                ),

            "Top10_Overlap_Fraction":
                len(
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
                ) / 10
        })

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(

        os.path.join(

            severity_dir,

            "micu_sicu_shap_summary.csv"
        ),

        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print(
        "============================================================"
    )

    print(
        f"MICU/SICU SHAP RESULTS — {severity}"
    )

    print(
        "============================================================"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # ========================================================
    # RETURN FOR JUPYTER
    # ========================================================

    if return_df:

        return summary


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

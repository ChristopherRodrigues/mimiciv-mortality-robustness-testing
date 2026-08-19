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
#
# Identical to MICU_SHAP.py / SICU_SHAP.py so every model
# pipeline and deep-model checkpoint is loaded from the same
# known-working locations.
# ============================================================

BASE_DIR = r"C:/Users/chris/Thesis/mimic4-benchmarks"

DATA_DIR = os.path.join(
    BASE_DIR,
    "data/in-hospital-mortality"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "weight_shap"
)

sys.path.append(
    os.path.join(
        BASE_DIR,
        "mimic4models"
    )
)


# ============================================================
# WEIGHT POPULATIONS
#
# Instead of perturbing one population (MICU/SICU), this
# script compares two disjoint real populations:
#
#   Reference:  60 kg <= weight < 90 kg
#   High:       weight >= 120 kg
# ============================================================

REFERENCE_WEIGHT_MIN = 60
REFERENCE_WEIGHT_MAX = 90
HIGH_WEIGHT_MIN = 120


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# WEIGHT EXTRACTION / POPULATION SAMPLING
# ============================================================

def extract_weight(X_raw, header):

    weight_idx = header.index("Weight")

    weights = []

    for patient in X_raw:

        values = np.asarray(patient)[:, weight_idx]

        values = pd.to_numeric(values, errors="coerce")

        valid = values[np.isfinite(values)]

        if len(valid) > 0:
            weights.append(valid[0])
        else:
            weights.append(np.nan)

    return np.asarray(weights, dtype=float)


def sample_weight_population(
    X,
    y,
    icu,
    weights,
    weight_min,
    weight_max=None
):

    mask = np.isfinite(weights)

    mask &= weights >= weight_min

    if weight_max is not None:
        mask &= weights < weight_max

    return (
        X[mask],
        np.asarray(y)[mask],
        np.asarray(icu)[mask],
        weights[mask]
    )


# ============================================================
# ICU MAP
# ============================================================

def build_icu_map(root_path):

    icu_map = {}

    for split in ["train", "test"]:

        split_path = os.path.join(root_path, split)

        if not os.path.exists(split_path):
            continue

        for subject_id in os.listdir(split_path):

            subject_path = os.path.join(split_path, subject_id)

            if not os.path.isdir(subject_path):
                continue

            stays_file = os.path.join(subject_path, "stays.csv")

            if not os.path.exists(stays_file):
                continue

            stays_df = pd.read_csv(stays_file)

            if "intime" in stays_df.columns:
                stays_df = stays_df.sort_values(by="intime")

            if "LAST_CAREUNIT" not in stays_df.columns:
                continue

            icu_units = stays_df["LAST_CAREUNIT"].tolist()

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
# PIPELINE
# ============================================================

def load_pipeline(path):

    return joblib.load(path)


# ============================================================
# CLASSICAL TRANSFORMATION
#
# Ported directly from MICU_SHAP.py. This is the piece the
# previous version of this script reimplemented independently
# — that mismatch against the actual pipeline .pkl schema was
# almost certainly the source of the model errors.
# ============================================================

def transform_classical(X, icu, pipeline):

    X = X[:, pipeline["non_empty_cols"]]

    X = pipeline["imputer"].transform(X)

    X = pipeline["variance_selector"].transform(X)

    if len(pipeline["correlated_columns_removed"]) > 0:

        drop = set(pipeline["correlated_columns_removed"])

        mask = np.array([
            f not in drop
            for f in pipeline["feature_names_after_variance"]
        ])

        X = X[:, mask]

    if (
        "scaler" in pipeline
        and pipeline["scaler"] is not None
    ):

        X = pipeline["scaler"].transform(X)

    icu_enc = pipeline["icu_encoder"].transform(
        np.asarray(icu).reshape(-1, 1)
    )

    X = np.hstack([X, icu_enc])

    return X


def get_classical_feature_names(pipeline):

    if "final_feature_names" in pipeline:

        return np.asarray(pipeline["final_feature_names"])

    feature_names = list(pipeline["feature_names_after_variance"])

    if len(pipeline["correlated_columns_removed"]) > 0:

        drop = set(pipeline["correlated_columns_removed"])

        feature_names = [
            f for f in feature_names if f not in drop
        ]

    try:

        encoder = pipeline["icu_encoder"]

        icu_names = encoder.get_feature_names_out(["ICU"])

        feature_names.extend(icu_names.tolist())

    except Exception:
        pass

    return np.asarray(feature_names)


# ============================================================
# MODEL UNWRAPPING / PROBABILITY WRAPPER / SHAP NORMALIZATION
# ============================================================

def unwrap_calibrated_model(model):

    if hasattr(model, "calibrated_classifiers_"):

        calibrated = model.calibrated_classifiers_

        if len(calibrated) > 0:

            inner = calibrated[0]

            if hasattr(inner, "estimator"):
                return inner.estimator

            if hasattr(inner, "base_estimator"):
                return inner.base_estimator

    return model


def model_probability_function(model):

    def predict(X):

        p = model.predict_proba(X)

        return p[:, 1]

    return predict


def normalize_shap_values(shap_values):

    if isinstance(shap_values, list):

        if len(shap_values) == 2:
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]

    return np.asarray(shap_values)


# ============================================================
# FEATURE IMPORTANCE (classical + deep)
# ============================================================

def mean_abs_shap(shap_values, feature_names):

    shap_values = normalize_shap_values(shap_values)

    if shap_values.ndim == 2:

        values = np.mean(np.abs(shap_values), axis=0)

    elif shap_values.ndim == 3:

        if shap_values.shape[2] == 2:

            shap_positive = shap_values[:, :, 1]

            values = np.mean(np.abs(shap_positive), axis=0)

        else:

            raise ValueError(
                f"Unsupported 3D SHAP shape: {shap_values.shape}"
            )

    else:

        raise ValueError(
            f"Unsupported SHAP shape: {shap_values.shape}"
        )

    feature_names = np.asarray(feature_names)

    if len(values) != len(feature_names):

        raise ValueError(
            "SHAP feature count does not match feature-name "
            f"count: {len(values)} vs {len(feature_names)}"
        )

    df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": values
    })

    df = df.sort_values(
        "Mean_Abs_SHAP", ascending=False
    ).reset_index(drop=True)

    df["Rank"] = np.arange(1, len(df) + 1)

    return df


def mean_abs_shap_deep(shap_values, feature_names):

    shap_values = np.asarray(shap_values)

    if shap_values.ndim != 3:

        raise ValueError(
            f"Deep SHAP values must have shape (N,T,F). "
            f"Got {shap_values.shape}"
        )

    values = np.mean(np.abs(shap_values), axis=(0, 1))

    feature_names = np.asarray(feature_names)

    if len(values) != len(feature_names):

        raise ValueError(
            "Deep SHAP feature count does not match "
            f"feature-name count: {len(values)} vs "
            f"{len(feature_names)}"
        )

    df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": values
    })

    df = df.sort_values(
        "Mean_Abs_SHAP", ascending=False
    ).reset_index(drop=True)

    df["Rank"] = np.arange(1, len(df) + 1)

    return df


# ============================================================
# PRINT SHAP RESULTS
#
# NOTE: column names below say "Perturbed" for consistency
# with MICU_SHAP.py / SICU_SHAP.py's proven compare_shap()
# implementation. In this script "Perturbed" always means
# "the high-weight (>=120kg) population" — there is no actual
# perturbation happening here, just a population comparison.
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

    print()
    print("Top 10 rows (by absolute SHAP change):")

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

    print(top10_display.to_string(index=False))

    top10_reference = set(reference_df.head(10)["Feature"])

    top10_perturbed = set(perturbed_df.head(10)["Feature"])

    overlap = top10_reference & top10_perturbed

    print(f"Top-10 feature overlap: {len(overlap)}/10")

    increasing = comparison[
        comparison["SHAP_Change"] > 0
    ].sort_values("SHAP_Change", ascending=False).head(10)

    print()
    print("Top features increasing under high weight:")

    if len(increasing) == 0:

        print("None")

    else:

        increasing_display = increasing[
            [
                "Feature",
                "Mean_Abs_SHAP_Reference",
                "Mean_Abs_SHAP_Perturbed"
            ]
        ].rename(columns={
            "Mean_Abs_SHAP_Reference": "Reference",
            "Mean_Abs_SHAP_Perturbed": "HighWeight"
        })

        print(increasing_display.to_string(index=False))

    print()
    print("Feature rank comparison:")

    rank_display = comparison.sort_values("Rank_Reference")[[
        "Feature", "Rank_Reference", "Rank_Perturbed"
    ]]

    print(rank_display.to_string(index=False))

    print()
    print("Feature SHAP change:")

    change_display = comparison.sort_values(
        "SHAP_Change", ascending=False
    )[["Feature", "SHAP_Change"]]

    print(change_display.to_string(index=False))


# ============================================================
# SHAP COMPARISON
#
# Ported unchanged from MICU_SHAP.py. Works for populations of
# different sizes since reference/perturbed are each aggregated
# to one Mean_Abs_SHAP value per feature before being merged.
# ============================================================

def compare_shap(
    reference_shap,
    perturbed_shap,
    feature_names,
    out_dir,
    suffix,
    deep=False
):

    os.makedirs(out_dir, exist_ok=True)

    if deep:

        reference_df = mean_abs_shap_deep(reference_shap, feature_names)

        perturbed_df = mean_abs_shap_deep(perturbed_shap, feature_names)

    else:

        reference_df = mean_abs_shap(reference_shap, feature_names)

        perturbed_df = mean_abs_shap(perturbed_shap, feature_names)

    reference_df.to_csv(
        os.path.join(out_dir, f"mean_abs_shap_reference_{suffix}.csv"),
        index=False
    )

    perturbed_df.to_csv(
        os.path.join(out_dir, f"mean_abs_shap_highweight_{suffix}.csv"),
        index=False
    )

    comparison = pd.merge(

        reference_df[["Feature", "Mean_Abs_SHAP", "Rank"]],

        perturbed_df[["Feature", "Mean_Abs_SHAP", "Rank"]],

        on="Feature",

        suffixes=("_Reference", "_Perturbed")
    )

    comparison["SHAP_Change"] = (
        comparison["Mean_Abs_SHAP_Perturbed"]
        - comparison["Mean_Abs_SHAP_Reference"]
    )

    comparison["Absolute_SHAP_Change"] = np.abs(comparison["SHAP_Change"])

    comparison["Rank_Change"] = (
        comparison["Rank_Perturbed"] - comparison["Rank_Reference"]
    )

    comparison = comparison.sort_values(
        "Absolute_SHAP_Change", ascending=False
    ).reset_index(drop=True)

    comparison.to_csv(
        os.path.join(out_dir, f"shap_comparison_{suffix}.csv"),
        index=False
    )

    merged_rank = comparison.sort_values("Feature")

    rho, p_value = spearmanr(
        merged_rank["Rank_Reference"], merged_rank["Rank_Perturbed"]
    )

    pd.DataFrame({
        "Spearman_Rho": [rho],
        "P_Value": [p_value]
    }).to_csv(
        os.path.join(out_dir, f"rank_correlation_{suffix}.csv"),
        index=False
    )

    plt.figure(figsize=(7, 7))

    plt.scatter(
        comparison["Rank_Reference"], comparison["Rank_Perturbed"], alpha=0.7
    )

    lim = len(feature_names)

    plt.plot([1, lim], [1, lim], linestyle="--")

    plt.xlabel("Reference (60-90kg) SHAP Rank")
    plt.ylabel("High-weight (>=120kg) SHAP Rank")

    plt.title(f"Weight SHAP Rank Stability\nSpearman \u03c1 = {rho:.3f}")

    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, f"rank_scatter_{suffix}.png"), dpi=300
    )

    plt.close()

    top10_reference = set(reference_df.head(10)["Feature"])

    top10_perturbed = set(perturbed_df.head(10)["Feature"])

    overlap = top10_reference & top10_perturbed

    pd.DataFrame({
        "Reference_Top10": [", ".join(reference_df.head(10)["Feature"])],
        "HighWeight_Top10": [", ".join(perturbed_df.head(10)["Feature"])],
        "Overlap_Count": [len(overlap)],
        "Overlap_Fraction": [len(overlap) / 10]
    }).to_csv(
        os.path.join(out_dir, f"top10_overlap_{suffix}.csv"), index=False
    )

    plot_df = comparison.sort_values(
        "SHAP_Change", ascending=False
    ).head(min(15, len(comparison))).copy()

    plot_df = plot_df.sort_values("SHAP_Change")

    plt.figure(figsize=(9, 7))

    plt.barh(plot_df["Feature"], plot_df["SHAP_Change"])

    plt.axvline(0, linestyle="--")

    plt.xlabel("Change in Mean |SHAP|")
    plt.ylabel("Feature")

    plt.title(
        "Weight Feature Attribution Change\n"
        "Reference (60-90kg) \u2192 High weight (>=120kg)"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, f"shap_change_{suffix}.png"), dpi=300
    )

    plt.close()

    print_shap_results(
        model_name=suffix,
        comparison=comparison,
        reference_df=reference_df,
        perturbed_df=perturbed_df,
        rho=rho,
        p_value=p_value
    )

    return (comparison, rho, p_value, reference_df, perturbed_df)


# ============================================================
# CLASSICAL SHAP
#
# Ported unchanged from MICU_SHAP.py.
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

    print(f"\nRunning SHAP: {model_name}")

    pipeline = load_pipeline(pipeline_path)

    model = pipeline["model"]

    X_ref = transform_classical(X_reference, icu_reference, pipeline)

    X_pert = transform_classical(X_perturbed, icu_perturbed, pipeline)

    feature_names = get_classical_feature_names(pipeline)

    if len(feature_names) != X_ref.shape[1]:

        raise RuntimeError(
            f"{model_name}: feature-name count "
            f"{len(feature_names)} does not match "
            f"transformed feature count {X_ref.shape[1]}"
        )

    explain_model = unwrap_calibrated_model(model)

    if model_name == "XGBoost":

        explainer = shap.TreeExplainer(explain_model)

        shap_ref = normalize_shap_values(explainer.shap_values(X_ref))

        shap_pert = normalize_shap_values(explainer.shap_values(X_pert))

    elif model_name == "LogisticRegression":

        background_size = min(100, len(X_ref))

        rng = np.random.default_rng(RANDOM_SEED)

        background_idx = rng.choice(
            len(X_ref), size=background_size, replace=False
        )

        background = X_ref[background_idx]

        explainer = shap.LinearExplainer(explain_model, background)

        shap_ref = normalize_shap_values(explainer.shap_values(X_ref))

        shap_pert = normalize_shap_values(explainer.shap_values(X_pert))

    elif model_name == "Random Forest":

        background_size = min(100, len(X_ref))

        rng = np.random.default_rng(RANDOM_SEED)

        background_idx = rng.choice(
            len(X_ref), size=background_size, replace=False
        )

        background = X_ref[background_idx]

        explainer = shap.TreeExplainer(explain_model, data=background)

        shap_ref = normalize_shap_values(explainer.shap_values(X_ref))

        shap_pert = normalize_shap_values(explainer.shap_values(X_pert))

    elif model_name == "MLP":

        background_size = min(100, len(X_ref))

        rng = np.random.default_rng(RANDOM_SEED)

        background_idx = rng.choice(
            len(X_ref), size=background_size, replace=False
        )

        background = X_ref[background_idx]

        explainer = shap.KernelExplainer(
            model_probability_function(explain_model), background
        )

        n_ref_explain = min(100, len(X_ref))

        n_pert_explain = min(100, len(X_pert))

        shap_ref = normalize_shap_values(
            explainer.shap_values(X_ref[:n_ref_explain], silent=True)
        )

        shap_pert = normalize_shap_values(
            explainer.shap_values(X_pert[:n_pert_explain], silent=True)
        )

    else:

        raise ValueError(f"Unsupported classical model: {model_name}")

    np.save(
        os.path.join(out_dir, f"shap_reference_{model_name}.npy"), shap_ref
    )

    np.save(
        os.path.join(out_dir, f"shap_highweight_{model_name}.npy"), shap_pert
    )

    suffix = model_name.lower().replace(" ", "_")

    (comparison, rho, p, reference_df, perturbed_df) = compare_shap(
        shap_ref, shap_pert, feature_names, out_dir, suffix, deep=False
    )

    return {
        "comparison": comparison,
        "rho": rho,
        "p": p,
        "reference_df": reference_df,
        "perturbed_df": perturbed_df
    }


# ============================================================
# DEEP MODEL LOADING
#
# Ported unchanged from MICU_SHAP.py.
# ============================================================

def load_lstm_module():

    module_path = os.path.join(
        BASE_DIR, "mimic4models", "keras_models", "lstm.py"
    )

    spec = importlib.util.spec_from_file_location("mimic_lstm", module_path)

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


def load_channelwise_module():

    module_path = os.path.join(
        BASE_DIR, "mimic4models", "keras_models", "channel_wise_lstms.py"
    )

    spec = importlib.util.spec_from_file_location(
        "mimic_channelwise", module_path
    )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


def build_deep_model(model_name, pipeline):

    if model_name in ["LSTM", "LSTM_DS"]:

        lstm_module = load_lstm_module()

        target_repl = (model_name == "LSTM_DS")

        return lstm_module.Network(
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

    if model_name in ["ChannelwiseLSTM", "ChannelwiseLSTM_DS"]:

        channelwise_module = load_channelwise_module()

        target_repl = (model_name == "ChannelwiseLSTM_DS")

        return channelwise_module.Network(
            dim=16,
            batch_norm=False,
            dropout=0.3,
            rec_dropout=0.0,
            header=pipeline["feature_names"],
            task="ihm",
            target_repl=target_repl,
            deep_supervision=False,
            depth=2,
            input_dim=96,
            size_coef=4.0
        )

    raise ValueError(f"Unsupported deep model: {model_name}")


# ============================================================
# DEEP SHAP — WEIGHT POPULATION VERSION
#
# Adapted from MICU_SHAP.py's run_deep_shap(). The key
# difference: MICU compares the SAME patients before/after a
# perturbation, so it requires X_reference and X_perturbed to
# have identical shape and samples them with paired indices.
# Here, "reference" and "high weight" are two DIFFERENT,
# differently-sized populations, so:
#   - only the (timesteps, features) dims are required to match
#   - reference and high-weight rows are sampled independently
# ============================================================

def run_deep_shap(
    model_name,
    model_path,
    pipeline_path,
    X_reference,
    X_high,
    out_dir
):

    os.makedirs(out_dir, exist_ok=True)

    pipeline = joblib.load(pipeline_path)

    feature_names = list(pipeline["feature_names"])

    model = build_deep_model(model_name, pipeline)

    model.load_weights(model_path)

    X_reference = np.asarray(X_reference, dtype=np.float32)

    X_high = np.asarray(X_high, dtype=np.float32)

    if X_reference.ndim != 3:

        raise ValueError(
            f"X_reference must have shape (N,T,F). Got {X_reference.shape}"
        )

    if X_high.ndim != 3:

        raise ValueError(
            f"X_high must have shape (N,T,F). Got {X_high.shape}"
        )

    if X_reference.shape[1:] != X_high.shape[1:]:

        raise ValueError(
            f"{model_name}: reference and high-weight tensors must "
            f"share the same timesteps/features. "
            f"Reference: {X_reference.shape}, High: {X_high.shape}"
        )

    n_timesteps = X_reference.shape[1]

    n_features = X_reference.shape[2]

    if len(feature_names) != n_features:

        raise ValueError(
            f"Number of feature names ({len(feature_names)}) does not "
            f"match input features ({n_features})."
        )

    print()
    print(
        f"{model_name} deep SHAP input shapes — "
        f"reference: {X_reference.shape}, high: {X_high.shape}"
    )

    rng = np.random.default_rng(RANDOM_SEED)

    n_background = min(50, len(X_reference))

    n_ref_explain = min(100, len(X_reference))

    n_high_explain = min(100, len(X_high))

    background = X_reference[
        rng.choice(len(X_reference), n_background, replace=False)
    ]

    reference_sample = X_reference[
        rng.choice(len(X_reference), n_ref_explain, replace=False)
    ]

    high_sample = X_high[
        rng.choice(len(X_high), n_high_explain, replace=False)
    ]

    print("SHAP background:", background.shape)
    print("SHAP reference:", reference_sample.shape)
    print("SHAP high-weight:", high_sample.shape)

    fixed_input = tf.keras.Input(
        shape=(n_timesteps, n_features),
        dtype=tf.float32,
        name=f"{model_name}_shap_input"
    )

    outputs = model(fixed_input, training=False)

    if isinstance(outputs, (list, tuple)):
        mortality_output = outputs[0]
    else:
        mortality_output = outputs

    if len(mortality_output.shape) != 2:

        raise ValueError(
            f"Mortality output must be (batch, output). "
            f"Got {mortality_output.shape}"
        )

    if mortality_output.shape[-1] != 1:

        raise ValueError(
            f"Expected one mortality output. Got {mortality_output.shape}"
        )

    fixed_model = tf.keras.Model(
        inputs=fixed_input,
        outputs=mortality_output,
        name=f"{model_name}_mortality_SHAP"
    )

    print("Fixed model input shape:", fixed_model.input_shape)
    print("Fixed model output shape:", fixed_model.output_shape)

    test_prediction = fixed_model.predict(background[:2], verbose=0)

    if test_prediction.shape != (2, 1):

        raise ValueError(
            f"Wrapped model does not produce expected (N,1) output. "
            f"Got {test_prediction.shape}"
        )

    print("\nCreating GradientExplainer...")

    explainer = shap.GradientExplainer(
        (fixed_input, mortality_output), background
    )

    print("GradientExplainer created successfully.")

    print("\nCalculating reference SHAP...")

    shap_reference_raw = explainer.shap_values(reference_sample, nsamples=200)

    print("Calculating high-weight SHAP...")

    shap_high_raw = explainer.shap_values(high_sample, nsamples=200)

    def process_deep_shap(values):

        if isinstance(values, list):

            if len(values) != 1:
                raise ValueError("Expected exactly one SHAP output.")

            values = values[0]

        values = np.asarray(values)

        if values.ndim == 4 and values.shape[-1] == 1:
            values = np.squeeze(values, axis=-1)

        if values.ndim != 3:

            raise ValueError(
                f"Expected deep SHAP shape (N,T,F). Got {values.shape}"
            )

        expected_shape = (values.shape[0], n_timesteps, n_features)

        if values.shape != expected_shape:

            raise ValueError(
                f"Unexpected deep SHAP shape. Expected {expected_shape}, "
                f"got {values.shape}"
            )

        return values

    shap_reference = process_deep_shap(shap_reference_raw)

    shap_high = process_deep_shap(shap_high_raw)

    print("Reference SHAP shape:", shap_reference.shape)
    print("High-weight SHAP shape:", shap_high.shape)

    suffix = model_name.lower()

    np.save(
        os.path.join(out_dir, f"shap_reference_{suffix}.npy"), shap_reference
    )

    np.save(
        os.path.join(out_dir, f"shap_highweight_{suffix}.npy"), shap_high
    )

    (comparison, rho, p, reference_df, perturbed_df) = compare_shap(
        shap_reference, shap_high, feature_names, out_dir, suffix, deep=True
    )

    reference_shap_feature = np.mean(np.abs(shap_reference), axis=1)

    high_shap_feature = np.mean(np.abs(shap_high), axis=1)

    reference_input_feature = np.mean(reference_sample, axis=1)

    high_input_feature = np.mean(high_sample, axis=1)

    plt.figure()

    shap.summary_plot(
        reference_shap_feature, reference_input_feature,
        feature_names=feature_names, show=False
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, f"summary_reference_{suffix}.png"),
        dpi=300, bbox_inches="tight"
    )

    plt.close()

    plt.figure()

    shap.summary_plot(
        high_shap_feature, high_input_feature,
        feature_names=feature_names, show=False
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, f"summary_highweight_{suffix}.png"),
        dpi=300, bbox_inches="tight"
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

def main(return_df=False, cli_args=None):

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--period", type=str, default="all",
        choices=[
            "first4days", "first8days", "last12hours",
            "first25percent", "first50percent", "all"
        ]
    )

    parser.add_argument(
        "--features", type=str, default="all",
        choices=["all", "len", "all_but_len"]
    )

    parser.add_argument(
        "--data", type=str, default=DATA_DIR
    )

    parser.add_argument(
        "--output_dir", type=str, default=RESULT_DIR
    )

    args = parser.parse_args(cli_args)

    data_dir = args.data

    # ========================================================
    # READ TEST DATA
    # ========================================================

    test_reader = InHospitalMortalityReader(
        dataset_dir=os.path.join(data_dir, "test"),
        listfile=os.path.join(data_dir, "test_listfile.csv"),
        period_length=48.0
    )

    icu_map = build_icu_map(os.path.join(data_dir, "../root"))

    print("Reading test data...")

    test_ret = common_utils.read_chunk(
        test_reader, test_reader.get_number_of_examples()
    )

    X_raw = test_ret["X"]

    y = np.asarray(test_ret["y"])

    names = test_ret["name"]

    feature_names = np.asarray(test_ret["header"])

    icu = np.asarray([
        icu_map.get(n, "UNKNOWN") for n in names
    ])

    # ========================================================
    # WEIGHT EXTRACTION
    # ========================================================

    weights = extract_weight(X_raw, test_ret["header"])

    valid_weights = weights[np.isfinite(weights)]

    print()
    print("Valid weights:", len(valid_weights))
    print("Mean weight:", np.mean(valid_weights))
    print("Min weight:", np.min(valid_weights))
    print("Max weight:", np.max(valid_weights))

    # ========================================================
    # CLASSICAL FEATURE EXTRACTION
    # ========================================================

    X_base = common_utils.extract_features_from_rawdata(
        X_raw, feature_names, args.period, args.features
    )

    (
        X_reference_classical, y_reference, icu_reference, w_reference
    ) = sample_weight_population(
        X_base, y, icu, weights,
        weight_min=REFERENCE_WEIGHT_MIN, weight_max=REFERENCE_WEIGHT_MAX
    )

    (
        X_high_classical, y_high, icu_high, w_high
    ) = sample_weight_population(
        X_base, y, icu, weights, weight_min=HIGH_WEIGHT_MIN
    )

    print()
    print(f"Reference population (60-90kg): {len(X_reference_classical)}")
    print(f"High-weight population (>=120kg): {len(X_high_classical)}")

    if len(X_reference_classical) == 0 or len(X_high_classical) == 0:

        raise RuntimeError(
            "One of the weight populations is empty — check the "
            "Weight column and thresholds."
        )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    os.makedirs(args.output_dir, exist_ok=True)

    # ========================================================
    # CLASSICAL MODEL PATHS — identical to MICU_SHAP.py
    # ========================================================

    classical_models = {

        "XGBoost": os.path.join(
            BASE_DIR, "mimic4models", "in_hospital_mortality",
            "XGBoostTuned", "xgb_pipeline.pkl"
        ),

        "LogisticRegression": os.path.join(
            BASE_DIR, "mimic4models", "in_hospital_mortality",
            "logistic", "logistic_pipeline.pkl"
        ),

        "Random Forest": os.path.join(
            BASE_DIR, "mimic4models", "in_hospital_mortality",
            "RandomForestTuned", "rf_pipeline.pkl"
        ),

        "MLP": os.path.join(
            BASE_DIR, "mimic4models", "in_hospital_mortality",
            "MLP", "mlp_pipeline.pkl"
        )
    }

    # ========================================================
    # CLASSICAL SHAP
    # ========================================================

    results = {}

    for model_name, pipeline_path in classical_models.items():

        print()
        print(f"Running classical SHAP: {model_name}")

        model_dir = os.path.join(
            args.output_dir, model_name.replace(" ", "_")
        )

        os.makedirs(model_dir, exist_ok=True)

        results[model_name] = run_classical_shap(
            model_name=model_name,
            pipeline_path=pipeline_path,
            X_reference=X_reference_classical,
            X_perturbed=X_high_classical,
            icu_reference=icu_reference,
            icu_perturbed=icu_high,
            out_dir=model_dir
        )

    # ========================================================
    # DEEP SNAPSHOT — identical source to MICU_SHAP.py
    #
    # NOTE: the snapshot's own y/icu are used for the deep
    # population split, while `weights` is the array computed
    # above from the raw test set. This assumes the snapshot's
    # patient order/count matches the raw test set — the same
    # assumption already relied on by the original version of
    # this script.
    # ========================================================

    snapshot_path = os.path.join(
        BASE_DIR, "mimic4models", "in_hospital_mortality",
        "LSTM", "test_snapshot.pkl"
    )

    snapshot = joblib.load(snapshot_path)

    X_norm = np.asarray(snapshot["X"], dtype=np.float32)

    snapshot_y = np.asarray(snapshot["y"])

    snapshot_icu = np.asarray(snapshot["icu"])

    if len(snapshot_icu) != len(weights):

        raise RuntimeError(
            "Deep snapshot row count "
            f"({len(snapshot_icu)}) does not match the raw test "
            f"set row count used for weight extraction "
            f"({len(weights)}). The two are assumed to be in the "
            "same patient order — this needs to be reconciled "
            "before the deep-model weight comparison can run."
        )

    (
        X_deep_reference, _, icu_deep_reference, _
    ) = sample_weight_population(
        X_norm, snapshot_y, snapshot_icu, weights,
        weight_min=REFERENCE_WEIGHT_MIN, weight_max=REFERENCE_WEIGHT_MAX
    )

    (
        X_deep_high, _, icu_deep_high, _
    ) = sample_weight_population(
        X_norm, snapshot_y, snapshot_icu, weights, weight_min=HIGH_WEIGHT_MIN
    )

    print()
    print(f"Deep reference population (60-90kg): {len(X_deep_reference)}")
    print(f"Deep high-weight population (>=120kg): {len(X_deep_high)}")

    if len(X_deep_reference) == 0 or len(X_deep_high) == 0:

        raise RuntimeError(
            "One of the deep-learning weight populations is empty."
        )

    # ========================================================
    # DEEP MODEL PATHS — identical to MICU_SHAP.py
    # ========================================================

    deep_models = {

        "LSTM": os.path.join(
            BASE_DIR, "mimic4models", "in_hospital_mortality",
            "keras_states",
            "k_lstm.n16.d0.3.dep2.bs8.ts1.0."
            "epoch95.test0.23363609611988068.keras"
        ),

        "LSTM_DS": os.path.join(
            BASE_DIR, "mimic4models", "in_hospital_mortality",
            "LSTM_DS", "keras_states",
            "k_lstm.n16.d0.3.dep2.bs8.ts1.0."
            "trc0.5.epoch69.test0.2358170449733734.keras"
        ),

        "ChannelwiseLSTM": os.path.join(
            BASE_DIR, "mimic4models", "in_hospital_mortality",
            "ChannelwiseLSTM", "keras_states",
            "k_channel_wise_lstms.n16."
            "szc4.0.d0.3.dep2.bs8.ts1.0."
            "epoch13.test0.2362385094165802.keras"
        ),

        "ChannelwiseLSTM_DS": os.path.join(
            BASE_DIR, "mimic4models", "in_hospital_mortality",
            "ChannelwiseLSTM_DS", "keras_states",
            "k_channel_wise_lstms.n16."
            "szc4.0.d0.3.dep2.bs8.ts1.0."
            "trc0.5.epoch59."
            "test0.2366071343421936.keras"
        )
    }

    deep_pipelines = {

        "LSTM": os.path.join(
            BASE_DIR, "mimic4models", "in_hospital_mortality",
            "LSTM", "preprocessing_pipeline.pkl"
        ),

        "LSTM_DS": os.path.join(
            BASE_DIR, "mimic4models", "in_hospital_mortality",
            "LSTM_DS", "preprocessing_pipeline.pkl"
        ),

        "ChannelwiseLSTM": os.path.join(
            BASE_DIR, "mimic4models", "in_hospital_mortality",
            "ChannelwiseLSTM", "preprocessing_pipeline.pkl"
        ),

        "ChannelwiseLSTM_DS": os.path.join(
            BASE_DIR, "mimic4models", "in_hospital_mortality",
            "ChannelwiseLSTM_DS", "preprocessing_pipeline.pkl"
        )
    }

    # ========================================================
    # RUN DEEP SHAP
    # ========================================================

    for model_name in deep_models:

        print()
        print(f"Running deep SHAP: {model_name}")

        pipeline = joblib.load(deep_pipelines[model_name])

        deep_feature_names = np.asarray(pipeline["feature_names"])

        if X_deep_reference.shape[2] != len(deep_feature_names):

            raise RuntimeError(
                f"{model_name}: snapshot feature count "
                f"{X_deep_reference.shape[2]} does not match "
                f"pipeline feature count {len(deep_feature_names)}"
            )

        model_dir = os.path.join(args.output_dir, model_name)

        os.makedirs(model_dir, exist_ok=True)

        results[model_name] = run_deep_shap(
            model_name=model_name,
            model_path=deep_models[model_name],
            pipeline_path=deep_pipelines[model_name],
            X_reference=X_deep_reference,
            X_high=X_deep_high,
            out_dir=model_dir
        )

    # ========================================================
    # COMBINED SUMMARY — same pattern as MICU_SHAP.py, built
    # directly from the in-memory reference/perturbed
    # (= high-weight) feature-importance tables.
    # ========================================================

    model_order = [
        "XGBoost", "LogisticRegression", "Random Forest", "MLP",
        "LSTM", "LSTM_DS", "ChannelwiseLSTM", "ChannelwiseLSTM_DS"
    ]

    summary_rows = []

    for model_name in model_order:

        result = results.get(model_name)

        if result is None:

            print(f"WARNING: no SHAP result found for {model_name}")

            continue

        overlap_count = len(
            set(result["reference_df"].head(10)["Feature"])
            & set(result["perturbed_df"].head(10)["Feature"])
        )

        summary_rows.append({
            "Model": model_name,
            "Comparison": "Reference (60-90kg) vs High (>=120kg)",
            "Spearman_Rho": result["rho"],
            "P_Value": result["p"],
            "Top10_Overlap": overlap_count,
            "Top10_Overlap_Fraction": overlap_count / 10
        })

    summary = pd.DataFrame(summary_rows)

    summary["Model"] = pd.Categorical(
        summary["Model"], categories=model_order, ordered=True
    )

    summary = summary.sort_values("Model").reset_index(drop=True)

    summary.to_csv(
        os.path.join(args.output_dir, "weight_shap_summary.csv"),
        index=False
    )

    print()
    print("============================================================")
    print("WEIGHT SHAP RESULTS")
    print("============================================================")

    print(summary.to_string(index=False))

    if return_df:
        return summary


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
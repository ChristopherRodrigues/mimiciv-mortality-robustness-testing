import os
import numpy as np
import pandas as pd
import argparse
import joblib
import sys
import copy
import seaborn as sns
import matplotlib.pyplot as plt

from mimic4benchmark.readers import InHospitalMortalityReader
from mimic4models import common_utils
from mimic4models.feature_extractor import build_feature_names
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
from tensorflow.keras.models import load_model

sys.path.append(r"C:/Users/chris/Thesis/mimic4-benchmarks/mimic4models")


# =========================
# UTILITIES
# =========================

# Values at or above this are treated as implausible / sentinel
# codes rather than real creatinine readings, and are dropped
# before cohort selection — mirroring the same precaution applied
# to the glucose sentinel-value issue (e.g. "9999" used as a
# missing-data marker in some MIMIC-IV chart data). Even the most
# extreme reported creatinine values in severe renal failure are
# well under this threshold, so it is set conservatively high to
# avoid discarding genuine extreme values.
MAX_PLAUSIBLE_CREATININE = 40.0


def extract_creatinine(X_raw, header):

    creat_idx = header.index("Creatinine")

    creatinine = []

    n_sentinel = 0

    for patient in X_raw:

        values = np.asarray(patient)[:, creat_idx]

        values = pd.to_numeric(
            values,
            errors="coerce"
        )

        # Drop implausible / sentinel-coded values before taking
        # the peak, so a sentinel code can never be chosen as a
        # patient's peak creatinine.
        sentinel_mask = values >= MAX_PLAUSIBLE_CREATININE

        n_sentinel += int(np.sum(sentinel_mask & np.isfinite(values)))

        values = np.where(sentinel_mask, np.nan, values)

        valid = values[np.isfinite(values)]

        if len(valid) > 0:
            # Peak creatinine reflects renal deterioration
            creatinine.append(np.max(valid))
        else:
            creatinine.append(np.nan)

    if n_sentinel > 0:

        print(
            f"Warning: dropped {n_sentinel} creatinine reading(s) "
            f">= {MAX_PLAUSIBLE_CREATININE} mg/dL as implausible/"
            f"sentinel values."
        )

    return np.asarray(creatinine, dtype=float)


def plot_creatinine_kde(
    c_all,
    c_reference,
    c_high,
    out_dir="creatinine_results"):

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    os.makedirs(out_dir, exist_ok=True)

    c_all = np.asarray(c_all, dtype=float)
    c_reference = np.asarray(c_reference, dtype=float)
    c_high = np.asarray(c_high, dtype=float)

    # ---------------------------------------------------------
    # Remove invalid values
    # ---------------------------------------------------------

    c_all = c_all[np.isfinite(c_all)]
    c_reference = c_reference[np.isfinite(c_reference)]
    c_high = c_high[np.isfinite(c_high)]

    # ---------------------------------------------------------
    # Calculate ONE KDE from the entire population
    # ---------------------------------------------------------

    kde = gaussian_kde(
        c_all,
        bw_method="scott"
    )

    # ---------------------------------------------------------
    # Evaluation range
    #
    # Use the 99th percentile as the upper plotting limit
    # to prevent extreme creatinine outliers from compressing
    # the clinically relevant distribution.
    # ---------------------------------------------------------

    lower_bound = max(
        0.0,
        c_all.min() - 0.1
    )

    upper_bound = np.percentile(
        c_all,
        99
    )

    # Small padding above the 99th percentile
    upper_bound += 0.1

    x = np.linspace(
        lower_bound,
        upper_bound,
        1000
    )

    y = kde(x)

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 6))

    # Entire population KDE
    plt.plot(
        x,
        y,
        linewidth=2,
        label=f"Entire population (n={len(c_all)})"
    )

    # Faded overall distribution
    plt.fill_between(
        x,
        y,
        alpha=0.08
    )

    # ---------------------------------------------------------
    # Reference creatinine region
    # 0.6 <= creatinine < 1.3 mg/dL
    # ---------------------------------------------------------

    reference_mask = (
        (x >= 0.6) &
        (x < 1.3)
    )

    plt.fill_between(
        x[reference_mask],
        y[reference_mask],
        alpha=0.30,
        label=(
            f"Reference creatinine "
            f"(0.6–<1.3 mg/dL, n={len(c_reference)})"
        )
    )

    # ---------------------------------------------------------
    # High creatinine region
    # creatinine >= 2.0 mg/dL
    # ---------------------------------------------------------

    high_mask = (
        x >= 2.0
    )

    plt.fill_between(
        x[high_mask],
        y[high_mask],
        alpha=0.30,
        label=(
            f"High creatinine "
            f"(≥2.0 mg/dL, n={len(c_high)})"
        )
    )

    # ---------------------------------------------------------
    # Cohort-selection boundaries
    # ---------------------------------------------------------

    plt.axvline(
        0.6,
        linestyle="--",
        linewidth=1.2,
        alpha=0.7
    )

    plt.axvline(
        1.3,
        linestyle="--",
        linewidth=1.2,
        alpha=0.7
    )

    plt.axvline(
        2.0,
        linestyle="--",
        linewidth=1.2,
        alpha=0.7
    )

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------

    plt.xlabel(
        "Peak creatinine (mg/dL)",
        fontsize=12
    )

    plt.ylabel(
        "Probability density",
        fontsize=12
    )

    plt.title(
        "Peak Creatinine Distribution with "
        "Reference and High-Creatinine Cohorts",
        fontsize=14
    )

    plt.legend(
        title="Population",
        fontsize=10
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    # Print percentile cutoff
    print(
        f"99th percentile creatinine: "
        f"{np.percentile(c_all, 99):.3f} mg/dL"
    )

    plt.tight_layout()

    output_path = os.path.join(
        out_dir,
        "creatinine_population_kde.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print(
        f"\nKDE plot saved to: {output_path}"
    )


def sample_creatinine_population(
        X,
        y,
        icu,
        creatinine,
        creat_min,
        creat_max=None):

    mask = np.isfinite(creatinine)

    mask &= creatinine >= creat_min

    if creat_max is not None:
        mask &= creatinine < creat_max

    return (
        X[mask],
        np.asarray(y)[mask],
        np.asarray(icu)[mask],
        creatinine[mask]
    )


# =========================
# DATA LOADING
# =========================

def build_icu_map(root_path):
    icu_map = {}
    for split in ["train", "test"]:
        split_path = os.path.join(root_path, split)
        for subject_id in os.listdir(split_path):
            subject_path = os.path.join(split_path, subject_id)
            if not os.path.isdir(subject_path):
                continue
            stays_file = os.path.join(subject_path, "stays.csv")
            if not os.path.exists(stays_file):
                continue
            stays_df = pd.read_csv(stays_file)
            stays_df = stays_df.sort_values(by="intime") if "intime" in stays_df.columns else stays_df
            icu_units = stays_df["LAST_CAREUNIT"].tolist()
            for i, icu in enumerate(icu_units):
                episode_idx = i + 1
                key = f"{subject_id}_episode{episode_idx}_timeseries.csv"
                icu_map[key] = icu
    return icu_map


def read_and_extract_features(reader, period, features, icu_map):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    y = ret['y']
    names = ret['name']
    icu = [icu_map.get(n, "UNKNOWN") for n in names]
    return (X, y, names, ret['header'], icu)


# =========================
# PIPELINE LOADER
# =========================

def load_pipeline(path):
    return joblib.load(path)


def load_raw(path):
    return joblib.load(path)


# =========================
# EVALUATION HELPERS
# =========================

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece, mce = 0.0, 0.0
    for i in range(n_bins):
        mask = binids == i
        if np.sum(mask) > 0:
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            gap = abs(acc - conf)
            ece += (np.sum(mask) / len(y_true)) * gap
            mce = max(mce, gap)
    return ece, mce


def get_probs(model, X):
    return model.predict_proba(X)[:, 1]


def evaluate(model, X, y):
    p = get_probs(model, X)
    y = np.asarray(y)
    ece, mce = expected_calibration_error(y, p)
    return {
        "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        "ECE": ece,
        "MCE": mce,
    }


def evaluate_deep(y_prob, y):
    y = np.asarray(y)
    ece, mce = expected_calibration_error(y, y_prob)
    return {
        "AUROC": roc_auc_score(y, y_prob),
        "AUPRC": average_precision_score(y, y_prob),
        "Brier": brier_score_loss(y, y_prob),
        "ECE": ece,
        "MCE": mce,
    }


# =========================
# BOOTSTRAP CONFIDENCE
# INTERVALS (INDEPENDENT
# SAMPLES)
# =========================

def bootstrap_metric_diff_ci_independent(
    y_reference,
    p_reference,
    y_high,
    p_high,
    n_bootstrap=2000,
    seed=42,
    alpha=0.05
):

    """
    Unpaired percentile bootstrap 95% CI for a
    (high-creatinine minus reference) metric difference.

    The two creatinine cohorts are disjoint, independently-
    selected groups of patients, not the same episodes before/
    after an edit, so this resamples each cohort independently —
    WITHOUT shared indices — rather than resampling one shared set
    of indices as the paired bootstrap used in the perturbation
    scripts (CCU, MICU, CVICU, MICU/SICU, SICU) does. This mirrors
    bootstrap_metric_diff_ci_independent in the glucose, FiO2 and
    weight/WBC sampling scripts.

    Draws are repeated until exactly n_bootstrap valid
    (both-class, in both cohorts) resamples are collected, so
    the returned interval is always based on the requested
    number of bootstrap replicates.
    """

    y_reference = np.asarray(y_reference)
    p_reference = np.asarray(p_reference)

    y_high = np.asarray(y_high)
    p_high = np.asarray(p_high)

    rng = np.random.default_rng(seed)

    n_ref = len(y_reference)
    n_high = len(y_high)

    diffs = {
        "AUROC": np.empty(n_bootstrap),
        "AUPRC": np.empty(n_bootstrap),
        "Brier": np.empty(n_bootstrap),
        "ECE": np.empty(n_bootstrap),
        "MCE": np.empty(n_bootstrap),
    }

    i = 0

    while i < n_bootstrap:

        idx_ref = rng.integers(0, n_ref, size=n_ref)
        idx_high = rng.integers(0, n_high, size=n_high)

        y_ref_b = y_reference[idx_ref]
        y_high_b = y_high[idx_high]

        # Each resampled cohort can occasionally
        # contain only one class.

        if np.unique(y_ref_b).size < 2:
            continue

        if np.unique(y_high_b).size < 2:
            continue

        p_ref_b = p_reference[idx_ref]
        p_high_b = p_high[idx_high]

        ece_ref, mce_ref = expected_calibration_error(y_ref_b, p_ref_b)
        ece_high, mce_high = expected_calibration_error(y_high_b, p_high_b)

        diffs["AUROC"][i] = (
            roc_auc_score(y_high_b, p_high_b)
            - roc_auc_score(y_ref_b, p_ref_b)
        )

        diffs["AUPRC"][i] = (
            average_precision_score(y_high_b, p_high_b)
            - average_precision_score(y_ref_b, p_ref_b)
        )

        diffs["Brier"][i] = (
            brier_score_loss(y_high_b, p_high_b)
            - brier_score_loss(y_ref_b, p_ref_b)
        )

        diffs["ECE"][i] = ece_high - ece_ref

        diffs["MCE"][i] = mce_high - mce_ref

        i += 1

    lower_q = 100 * (alpha / 2)
    upper_q = 100 * (1 - alpha / 2)

    ci = {}

    for metric, values in diffs.items():

        ci[f"Delta_{metric}_CI_lower"] = np.percentile(values, lower_q)
        ci[f"Delta_{metric}_CI_upper"] = np.percentile(values, upper_q)

    return ci


# =========================
# CORE EXPERIMENT
# =========================

def run_experiment(
        pipeline_path,
        X_reference,
        y_reference,
        icu_reference,
        X_high_creatinine,
        y_high_creatinine,
        icu_high_creatinine,
        out_dir=".",
        bootstrap_reps=2000):

    os.makedirs(out_dir, exist_ok=True)

    pipeline = load_pipeline(pipeline_path)
    model = pipeline["model"]

    def transform(X_in, icu_in):

        X_in = X_in[:, pipeline["non_empty_cols"]]

        X_in = pipeline["imputer"].transform(X_in)

        X_in = pipeline["variance_selector"].transform(X_in)

        if len(pipeline["correlated_columns_removed"]) > 0:

            drop = set(pipeline["correlated_columns_removed"])

            mask = np.array([
                f not in drop
                for f in pipeline["feature_names_after_variance"]
            ])

            X_in = X_in[:, mask]

        if "scaler" in pipeline and pipeline["scaler"] is not None:
            X_in = pipeline["scaler"].transform(X_in)

        icu_enc = pipeline["icu_encoder"].transform(
            np.asarray(icu_in).reshape(-1, 1)
        )

        X_in = np.hstack([X_in, icu_enc])

        return X_in

    # Transform both creatinine-defined cohorts independently
    X_reference_t = transform(
        X_reference,
        icu_reference
    )

    X_high_creatinine_t = transform(
        X_high_creatinine,
        icu_high_creatinine
    )

    p_reference = get_probs(model, X_reference_t)
    p_high_creatinine = get_probs(model, X_high_creatinine_t)

    # Evaluate
    reference_metrics = evaluate(
        model,
        X_reference_t,
        y_reference
    )

    high_creatinine_metrics = evaluate(
        model,
        X_high_creatinine_t,
        y_high_creatinine
    )

    ci = bootstrap_metric_diff_ci_independent(
        y_reference, p_reference,
        y_high_creatinine, p_high_creatinine,
        n_bootstrap=bootstrap_reps
    )

    return {
        "reference": reference_metrics,
        "high_creatinine": high_creatinine_metrics,
        "ci": ci,
    }


# =========================
# MULTI-MODEL RUNNER (CLASSICAL)
# =========================

def run_all_models(
        models,
        X_reference,
        y_reference,
        icu_reference,
        X_high_creatinine,
        y_high_creatinine,
        icu_high_creatinine,
        out_dir=".",
        bootstrap_reps=2000):

    os.makedirs(out_dir, exist_ok=True)

    results = []

    for name, path in models.items():

        print(f"Running {name}...")

        res = run_experiment(
            pipeline_path=path,

            X_reference=X_reference,
            y_reference=y_reference,
            icu_reference=icu_reference,

            X_high_creatinine=X_high_creatinine,
            y_high_creatinine=y_high_creatinine,
            icu_high_creatinine=icu_high_creatinine,

            out_dir=out_dir,
            bootstrap_reps=bootstrap_reps
        )

        reference = res["reference"]
        high_creatinine = res["high_creatinine"]

        results.append({

            "Model": name,

            # Reference creatinine cohort
            "Reference_AUROC": reference["AUROC"],
            "Reference_AUPRC": reference["AUPRC"],
            "Reference_Brier": reference["Brier"],
            "Reference_ECE": reference["ECE"],
            "Reference_MCE": reference["MCE"],

            # High creatinine cohort
            "HighCreatinine_AUROC": high_creatinine["AUROC"],
            "HighCreatinine_AUPRC": high_creatinine["AUPRC"],
            "HighCreatinine_Brier": high_creatinine["Brier"],
            "HighCreatinine_ECE": high_creatinine["ECE"],
            "HighCreatinine_MCE": high_creatinine["MCE"],

            # Difference (High creatinine − Reference)
            "Delta_AUROC": high_creatinine["AUROC"] - reference["AUROC"],
            "Delta_AUPRC": high_creatinine["AUPRC"] - reference["AUPRC"],
            "Delta_Brier": high_creatinine["Brier"] - reference["Brier"],
            "Delta_ECE": high_creatinine["ECE"] - reference["ECE"],
            "Delta_MCE": high_creatinine["MCE"] - reference["MCE"],

            **res["ci"],
        })

    df = pd.DataFrame(results)

    df.to_csv(
        os.path.join(out_dir, "creatinine_population_summary.csv"),
        index=False
    )

    return df


def run_all_models_deep(
        models: dict,
        snapshot_path,
        pipeline_map: dict,
        creatinine,
        out_dir=".",
        bootstrap_reps=2000):

    os.makedirs(out_dir, exist_ok=True)
    results = []

    snapshot = joblib.load(snapshot_path)

    X = snapshot["X"]
    y = np.asarray(snapshot["y"])
    icu = np.asarray(snapshot["icu"])

    # -----------------------------
    # Build the same creatinine cohorts
    # -----------------------------
    (
        X_reference,
        y_reference,
        _,
        _
    ) = sample_creatinine_population(
        X,
        y,
        icu,
        creatinine,
        creat_min=0.6,
        creat_max=1.3
    )

    (
        X_high_creatinine,
        y_high_creatinine,
        _,
        _
    ) = sample_creatinine_population(
        X,
        y,
        icu,
        creatinine,
        creat_min=2.0
    )

    for name, model_path in models.items():

        print(f"Running deep model: {name}...")

        pipeline = load_pipeline(pipeline_map[name])

        if name == "LSTM":
            from mimic4models.keras_models.lstm import Network

            model = Network(
                dim=16,
                batch_norm=False,
                dropout=0.3,
                rec_dropout=0.0,
                task="ihm",
                target_repl=False,
                deep_supervision=False,
                num_classes=1,
                depth=2,
                input_dim=96
            )

        elif name == "LSTM_DS":
            from mimic4models.keras_models.lstm import Network

            model = Network(
                dim=16,
                batch_norm=False,
                dropout=0.3,
                rec_dropout=0.0,
                task="ihm",
                target_repl=True,
                deep_supervision=False,
                num_classes=1,
                depth=2,
                input_dim=96
            )

        elif name == "ChannelwiseLSTM":
            from mimic4models.keras_models.channel_wise_lstms import Network

            model = Network(
                dim=16,
                batch_norm=False,
                dropout=0.3,
                rec_dropout=0.0,
                header=pipeline["feature_names"],
                task="ihm",
                target_repl=False,
                deep_supervision=False,
                depth=2,
                input_dim=96,
                size_coef=4.0
            )

        elif name == "ChannelwiseLSTM_DS":
            from mimic4models.keras_models.channel_wise_lstms import Network

            model = Network(
                dim=16,
                batch_norm=False,
                dropout=0.3,
                rec_dropout=0.0,
                header=pipeline["feature_names"],
                task="ihm",
                target_repl=True,
                deep_supervision=False,
                depth=2,
                input_dim=96,
                size_coef=4.0
            )

        model.load_weights(model_path)

        # -----------------------------
        # Predictions
        # -----------------------------
        p_reference = model.predict(X_reference)

        if isinstance(p_reference, list):
            p_reference = p_reference[0]

        p_reference = np.asarray(p_reference).ravel()

        p_high_creatinine = model.predict(X_high_creatinine)

        if isinstance(p_high_creatinine, list):
            p_high_creatinine = p_high_creatinine[0]

        p_high_creatinine = np.asarray(p_high_creatinine).ravel()

        # -----------------------------
        # Metrics
        # -----------------------------
        reference_metrics = evaluate_deep(
            p_reference,
            y_reference
        )

        high_creatinine_metrics = evaluate_deep(
            p_high_creatinine,
            y_high_creatinine
        )

        ci = bootstrap_metric_diff_ci_independent(
            y_reference, p_reference,
            y_high_creatinine, p_high_creatinine,
            n_bootstrap=bootstrap_reps
        )

        results.append({

            "Model": name,

            "Reference_AUROC": reference_metrics["AUROC"],
            "Reference_AUPRC": reference_metrics["AUPRC"],
            "Reference_Brier": reference_metrics["Brier"],
            "Reference_ECE": reference_metrics["ECE"],
            "Reference_MCE": reference_metrics["MCE"],

            "HighCreatinine_AUROC": high_creatinine_metrics["AUROC"],
            "HighCreatinine_AUPRC": high_creatinine_metrics["AUPRC"],
            "HighCreatinine_Brier": high_creatinine_metrics["Brier"],
            "HighCreatinine_ECE": high_creatinine_metrics["ECE"],
            "HighCreatinine_MCE": high_creatinine_metrics["MCE"],

            "Delta_AUROC": high_creatinine_metrics["AUROC"] - reference_metrics["AUROC"],
            "Delta_AUPRC": high_creatinine_metrics["AUPRC"] - reference_metrics["AUPRC"],
            "Delta_Brier": high_creatinine_metrics["Brier"] - reference_metrics["Brier"],
            "Delta_ECE": high_creatinine_metrics["ECE"] - reference_metrics["ECE"],
            "Delta_MCE": high_creatinine_metrics["MCE"] - reference_metrics["MCE"],

            **ci,
        })

    df = pd.DataFrame(results)

    df.to_csv(
        os.path.join(out_dir, "creatinine_deep_summary.csv"),
        index=False
    )

    return df


# =========================
# MAIN
# =========================

def main(return_df=False, cli_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', type=str, default='all',
                        choices=['first4days', 'first8days', 'last12hours',
                                 'first25percent', 'first50percent', 'all'])
    parser.add_argument('--features', type=str, default='all',
                        choices=['all', 'len', 'all_but_len'])
    parser.add_argument('--data', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             '../../../data/in-hospital-mortality/'))
    parser.add_argument('--output_dir', type=str, default='.')

    # ==================================================
    # NUMBER OF BOOTSTRAP REPETITIONS
    # ==================================================

    parser.add_argument(
        '--bootstrap_reps',
        type=int,
        default=2000,
        help='Number of independent-sample bootstrap repetitions'
    )

    args = parser.parse_args(cli_args)
    print(args)

    bootstrap_reps = args.bootstrap_reps

    print(f"Bootstrap repetitions: {bootstrap_reps}")

    test_reader = InHospitalMortalityReader(
        dataset_dir=os.path.join(args.data, 'test'),
        listfile=os.path.join(args.data, 'test_listfile.csv'),
        period_length=48.0
    )
    icu_map = build_icu_map(os.path.join(args.data, "../root"))

    print('Reading test data and extracting features ...')
    test_ret = common_utils.read_chunk(test_reader, test_reader.get_number_of_examples())
    test_icu = [icu_map.get(n, "UNKNOWN") for n in test_ret['name']]
    X_raw = test_ret['X']
    y = np.asarray(test_ret['y'])
    icu = test_icu
    feature_names = test_ret['header']

    def extract_static_variable(X_raw, idx):
        values = []

        for patient in X_raw:

            patient_values = np.asarray(patient)[:, idx]

            # Convert strings/objects to float
            patient_values = pd.to_numeric(
                patient_values,
                errors="coerce"
            )

            valid = patient_values[np.isfinite(patient_values)]

            if len(valid) > 0:
                values.append(np.max(valid))
            else:
                values.append(np.nan)

        return np.asarray(values, dtype=float)

    creatinine = extract_creatinine(
        X_raw,
        test_ret["header"]
    )

    print("Patients:", len(creatinine))
    print("Valid creatinine:", np.sum(np.isfinite(creatinine)))

    print(pd.Series(creatinine).describe())
    pd.Series(creatinine).describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95])

    # BASELINE features (classical)
    X_base = common_utils.extract_features_from_rawdata(
        X_raw, test_ret['header'], args.period, args.features
    )

    X_reference, y_reference, icu_reference, c_reference = sample_creatinine_population(
        X_base,
        y,
        icu,
        creatinine,
        creat_min=0.6,
        creat_max=1.3
    )

    X_high_creatinine, y_high_creatinine, icu_high_creatinine, c_high = sample_creatinine_population(
        X_base,
        y,
        icu,
        creatinine,
        creat_min=2.0
    )

    # ==================================================
    # OUTPUT ROOT
    # ==================================================

    result_dir = os.path.abspath(args.output_dir)

    os.makedirs(result_dir, exist_ok=True)

    print(f"Output directory: {result_dir}")

    # =========================
    # PLOT CREATININE DISTRIBUTIONS
    # =========================

    plot_creatinine_kde(
        c_all=creatinine,
        c_reference=c_reference,
        c_high=c_high,
        out_dir=result_dir
    )

    models = {
        "XGBoost": r"C:/Users/chris/Thesis/mimic4-benchmarks/mimic4models/in_hospital_mortality/XGBoostTuned/xgb_pipeline.pkl",
        "LogisticRegression": r"C:/Users/chris/Thesis/mimic4-benchmarks/mimic4models/in_hospital_mortality/logistic/logistic_pipeline.pkl",
        "Random Forest": r"C:/Users/chris/Thesis/mimic4-benchmarks/mimic4models/in_hospital_mortality/RandomForestTuned/rf_pipeline.pkl",
        "MLP": r"C:/Users/chris/Thesis/mimic4-benchmarks/mimic4models/in_hospital_mortality/MLP/mlp_pipeline.pkl"
    }

    deep_models = {
        "LSTM": r"C:/Users/chris/Thesis/mimic4-benchmarks/mimic4models/in_hospital_mortality/keras_states/k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch95.test0.23363609611988068.keras",
        "LSTM_DS": r"C:/Users/chris/Thesis/mimic4-benchmarks/mimic4models/in_hospital_mortality/LSTM_DS/keras_states/k_lstm.n16.d0.3.dep2.bs8.ts1.0.trc0.5.epoch69.test0.2358170449733734.keras",
        "ChannelwiseLSTM": r"C:/Users/chris/Thesis/mimic4-benchmarks/mimic4models/in_hospital_mortality/ChannelwiseLSTM/keras_states/k_channel_wise_lstms.n16.szc4.0.d0.3.dep2.bs8.ts1.0.epoch13.test0.2362385094165802.keras",
        "ChannelwiseLSTM_DS": r"C:/Users/chris/Thesis/mimic4-benchmarks/mimic4models/in_hospital_mortality/ChannelwiseLSTM_DS/keras_states/k_channel_wise_lstms.n16.szc4.0.d0.3.dep2.bs8.ts1.0.trc0.5.epoch59.test0.2366071343421936.keras"
    }

    print("\nReference creatinine cohort")
    print(f"N = {len(y_reference)}")
    print(f"Mean creatinine = {np.mean(c_reference):.2f}")
    print(f"Range = {c_reference.min():.2f} - {c_reference.max():.2f}")
    print(f"Mortality = {np.mean(y_reference):.3f}")

    print("\nHigh-creatinine cohort")
    print(f"N = {len(y_high_creatinine)}")
    print(f"Mean creatinine = {np.mean(c_high):.2f}")
    print(f"Range = {c_high.min():.2f} - {c_high.max():.2f}")
    print(f"Mortality = {np.mean(y_high_creatinine):.3f}")

    df_ml = run_all_models(
        models=models,

        X_reference=X_reference,
        y_reference=y_reference,
        icu_reference=icu_reference,

        X_high_creatinine=X_high_creatinine,
        y_high_creatinine=y_high_creatinine,
        icu_high_creatinine=icu_high_creatinine,

        out_dir=result_dir,
        bootstrap_reps=bootstrap_reps
    )
    df_ml["Family"] = "Classical"

    df_dl = run_all_models_deep(
        models=deep_models,
        snapshot_path=r"C:/Users/chris/Thesis/mimic4-benchmarks/mimic4models/in_hospital_mortality/LSTM/test_snapshot.pkl",
        pipeline_map={
            "LSTM": r"C:/Users/chris/Thesis/mimic4-benchmarks/mimic4models/in_hospital_mortality/LSTM/preprocessing_pipeline.pkl",
            "LSTM_DS": r"C:/Users/chris/Thesis/mimic4-benchmarks/mimic4models/in_hospital_mortality/LSTM_DS/preprocessing_pipeline.pkl",
            "ChannelwiseLSTM": r"C:/Users/chris/Thesis/mimic4-benchmarks/mimic4models/in_hospital_mortality/ChannelwiseLSTM/preprocessing_pipeline.pkl",
            "ChannelwiseLSTM_DS": r"C:/Users/chris/Thesis/mimic4-benchmarks/mimic4models/in_hospital_mortality/ChannelwiseLSTM_DS/preprocessing_pipeline.pkl"
        },
        creatinine=creatinine,
        out_dir=result_dir,
        bootstrap_reps=bootstrap_reps
    )

    df_dl["Family"] = "Deep"

    df = pd.concat([df_ml, df_dl], ignore_index=True)
    final_df = df

    final_df.to_csv(
        os.path.join(result_dir, "creatinine_summary.csv"),
        index=False
    )

    print("\nCombined Results:")
    print(final_df)

    if return_df:
        return final_df


if __name__ == "__main__":
    main()
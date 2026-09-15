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

def extract_fio2(X_raw, header):

    fio2_idx = header.index("Fraction inspired oxygen")

    fio2_values = []

    for patient in X_raw:

        values = np.asarray(patient)[:, fio2_idx]

        values = pd.to_numeric(
            values,
            errors="coerce"
        )

        valid = values[np.isfinite(values)]

        if len(valid) > 0:
            # First recorded FiO2 value
            fio2_values.append(valid[0])
        else:
            fio2_values.append(np.nan)

    return np.asarray(fio2_values, dtype=float)


def plot_fio2_kde(
        f_all,
        f_reference,
        f_high_fio2,
        out_dir="fio2_results"):

    os.makedirs(out_dir, exist_ok=True)

    f_all = np.asarray(f_all, dtype=float)
    f_reference = np.asarray(f_reference, dtype=float)
    f_high_fio2 = np.asarray(f_high_fio2, dtype=float)

    # Remove invalid values
    f_all = f_all[np.isfinite(f_all)]
    f_reference = f_reference[np.isfinite(f_reference)]
    f_high_fio2 = f_high_fio2[np.isfinite(f_high_fio2)]

    # ---------------------------------------------------------
    # Calculate ONE KDE from the entire population
    # ---------------------------------------------------------

    from scipy.stats import gaussian_kde

    kde = gaussian_kde(
        f_all,
        bw_method="scott"
    )

    # Evaluation range
    x_min = max(0.0, f_all.min() - 0.02)
    x_max = min(1.2, f_all.max() + 0.02)

    x = np.linspace(
        x_min,
        x_max,
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
        label=f"Entire population (n={len(f_all)})"
    )

    # Faded overall distribution
    plt.fill_between(
        x,
        y,
        alpha=0.08
    )

    # ---------------------------------------------------------
    # Shade REFERENCE portion of the SAME KDE
    # 0.21 <= FiO2 < 0.50
    # ---------------------------------------------------------

    reference_mask = (
        (x >= 0.21) &
        (x < 0.50)
    )

    plt.fill_between(
        x[reference_mask],
        y[reference_mask],
        alpha=0.30,
        label=(
            f"Reference FiO2 "
            f"(0.21–<0.50, n={len(f_reference)})"
        )
    )

    # ---------------------------------------------------------
    # Shade HIGH FiO2 portion of the SAME KDE
    # FiO2 >= 0.50
    # ---------------------------------------------------------

    high_mask = (
        x >= 0.50
    )

    plt.fill_between(
        x[high_mask],
        y[high_mask],
        alpha=0.30,
        label=(
            f"High FiO2 "
            f"(≥0.50, n={len(f_high_fio2)})"
        )
    )

    # ---------------------------------------------------------
    # Cohort boundaries
    # ---------------------------------------------------------

    plt.axvline(
        0.21,
        linestyle="--",
        linewidth=1.2,
        alpha=0.7
    )

    plt.axvline(
        0.50,
        linestyle="--",
        linewidth=1.2,
        alpha=0.7
    )

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------

    plt.xlabel(
        "Fraction inspired oxygen (FiO2)",
        fontsize=12
    )

    plt.ylabel(
        "Probability density",
        fontsize=12
    )

    plt.title(
        "FiO2 Distribution with Reference and High-FiO2 Cohorts",
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

    plt.tight_layout()

    output_path = os.path.join(
        out_dir,
        "fio2_population_kde.png"
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

def sample_fio2_population(
        X,
        y,
        icu,
        fio2_values,
        fio2_min,
        fio2_max=None):

    mask = np.isfinite(fio2_values)

    mask &= fio2_values >= fio2_min

    if fio2_max is not None:
        mask &= fio2_values < fio2_max

    return (
        X[mask],
        np.asarray(y)[mask],
        np.asarray(icu)[mask],
        fio2_values[mask]
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
            acc  = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            gap  = abs(acc - conf)
            ece += (np.sum(mask) / len(y_true)) * gap
            mce  = max(mce, gap)
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
        "ECE":   ece,
        "MCE":   mce,
    }

def evaluate_deep(y_prob, y):
    y = np.asarray(y)
    ece, mce = expected_calibration_error(y, y_prob)
    return {
        "AUROC": roc_auc_score(y, y_prob),
        "AUPRC": average_precision_score(y, y_prob),
        "Brier": brier_score_loss(y, y_prob),
        "ECE":   ece,
        "MCE":   mce,
    }

# =========================
# CORE EXPERIMENT
# =========================

def run_experiment(
        pipeline_path,
        X_reference,
        y_reference,
        icu_reference,
        X_high_fio2,
        y_high_fio2,
        icu_high_fio2,
        out_dir="."):

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

    # Transform both FiO2-defined cohorts independently
    X_reference_t = transform(
        X_reference,
        icu_reference
    )

    X_high_fio2_t = transform(
        X_high_fio2,
        icu_high_fio2
    )

    # Evaluate
    reference_metrics = evaluate(
        model,
        X_reference_t,
        y_reference
    )

    high_fio2_metrics = evaluate(
        model,
        X_high_fio2_t,
        y_high_fio2
    )

    return {
        "reference": reference_metrics,
        "high_fio2": high_fio2_metrics
    }
# =========================
# MULTI-MODEL RUNNER (CLASSICAL)
# =========================

def run_all_models(
        models,
        X_reference,
        y_reference,
        icu_reference,
        X_high_fio2,
        y_high_fio2,
        icu_high_fio2,
        out_dir="."):

    os.makedirs(out_dir, exist_ok=True)

    results = []

    for name, path in models.items():

        print(f"Running {name}...")

        res = run_experiment(
            pipeline_path=path,

            X_reference=X_reference,
            y_reference=y_reference,
            icu_reference=icu_reference,

            X_high_fio2=X_high_fio2,
            y_high_fio2=y_high_fio2,
            icu_high_fio2=icu_high_fio2,

            out_dir=out_dir
        )

        reference = res["reference"]
        high_fio2 = res["high_fio2"]

        results.append({

            "Model": name,

            # Reference FiO2 cohort
            "Reference_AUROC": reference["AUROC"],
            "Reference_AUPRC": reference["AUPRC"],
            "Reference_Brier": reference["Brier"],
            "Reference_ECE": reference["ECE"],
            "Reference_MCE": reference["MCE"],

            # High FiO2 cohort
            "HighFiO2_AUROC": high_fio2["AUROC"],
            "HighFiO2_AUPRC": high_fio2["AUPRC"],
            "HighFiO2_Brier": high_fio2["Brier"],
            "HighFiO2_ECE": high_fio2["ECE"],
            "HighFiO2_MCE": high_fio2["MCE"],

            # Difference (High FiO2 − Reference)
            "Delta_AUROC": high_fio2["AUROC"] - reference["AUROC"],
            "Delta_AUPRC": high_fio2["AUPRC"] - reference["AUPRC"],
            "Delta_Brier": high_fio2["Brier"] - reference["Brier"],
            "Delta_ECE": high_fio2["ECE"] - reference["ECE"],
            "Delta_MCE": high_fio2["MCE"] - reference["MCE"]
        })

    df = pd.DataFrame(results)

    df.to_csv(
        os.path.join(out_dir, "fio2_population_summary.csv"),
        index=False
    )

    return df
def run_all_models_deep(
        models: dict,
        snapshot_path,
        pipeline_map: dict,
        fio2_values,
        out_dir="."):

    os.makedirs(out_dir, exist_ok=True)
    results = []

    snapshot = joblib.load(snapshot_path)

    X = snapshot["X"]
    y = np.asarray(snapshot["y"])
    icu = np.asarray(snapshot["icu"])

    # -----------------------------
    # Build the same FiO2 cohorts
    # -----------------------------
    (
        X_reference,
        y_reference,
        _,
        _
    ) = sample_fio2_population(
        X,
        y,
        icu,
        fio2_values,
        fio2_min=0.21,
        fio2_max=0.50
    )

    (
        X_high_fio2,
        y_high_fio2,
        _,
        _
    ) = sample_fio2_population(
        X,
        y,
        icu,
        fio2_values,
        fio2_min=0.50
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

        p_high_fio2 = model.predict(X_high_fio2)

        if isinstance(p_high_fio2, list):
            p_high_fio2 = p_high_fio2[0]

        p_high_fio2 = np.asarray(p_high_fio2).ravel()

        # -----------------------------
        # Metrics
        # -----------------------------
        reference_metrics = evaluate_deep(
            p_reference,
            y_reference
        )

        high_fio2_metrics = evaluate_deep(
            p_high_fio2,
            y_high_fio2
        )

        results.append({

            "Model": name,

            "Reference_AUROC": reference_metrics["AUROC"],
            "Reference_AUPRC": reference_metrics["AUPRC"],
            "Reference_Brier": reference_metrics["Brier"],
            "Reference_ECE": reference_metrics["ECE"],
            "Reference_MCE": reference_metrics["MCE"],

            "HighFiO2_AUROC": high_fio2_metrics["AUROC"],
            "HighFiO2_AUPRC": high_fio2_metrics["AUPRC"],
            "HighFiO2_Brier": high_fio2_metrics["Brier"],
            "HighFiO2_ECE": high_fio2_metrics["ECE"],
            "HighFiO2_MCE": high_fio2_metrics["MCE"],

            "Delta_AUROC": high_fio2_metrics["AUROC"] - reference_metrics["AUROC"],
            "Delta_AUPRC": high_fio2_metrics["AUPRC"] - reference_metrics["AUPRC"],
            "Delta_Brier": high_fio2_metrics["Brier"] - reference_metrics["Brier"],
            "Delta_ECE": high_fio2_metrics["ECE"] - reference_metrics["ECE"],
            "Delta_MCE": high_fio2_metrics["MCE"] - reference_metrics["MCE"]
        })

    df = pd.DataFrame(results)

    df.to_csv(
        os.path.join(out_dir, "fio2_deep_summary.csv"),
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
    args = parser.parse_args(cli_args)
    print(args)


    test_reader = InHospitalMortalityReader(
        dataset_dir=os.path.join(args.data, 'test'),
        listfile=os.path.join(args.data, 'test_listfile.csv'),
        period_length=48.0
    )
    icu_map = build_icu_map(os.path.join(args.data, "../root"))

    print('Reading test data and extracting features ...')
    test_ret = common_utils.read_chunk(test_reader, test_reader.get_number_of_examples())
    test_icu = [icu_map.get(n, "UNKNOWN") for n in test_ret['name']]
    X_raw         = test_ret['X']
    #y             = test_ret['y']
    y = np.asarray(test_ret['y'])
    icu           = test_icu
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


    fio2_values = extract_fio2(
        X_raw,
        test_ret["header"]
    )

    print("Patients:", len(fio2_values))
    print("Valid fio2_values:", np.sum(np.isfinite(fio2_values)))

    print(pd.Series(fio2_values).describe())
    pd.Series(fio2_values).describe(percentiles=[0.25,0.5,0.75,0.9,0.95])
    # BASELINE features (classical)
    X_base = common_utils.extract_features_from_rawdata(
        X_raw, test_ret['header'], args.period, args.features
    )
    
    X_reference, y_reference, icu_reference, f_reference = sample_fio2_population(
        X_base,
        y,
        icu,
        fio2_values,
        fio2_min=0.21,
        fio2_max=0.50
    )

    X_high_fio2, y_high_fio2, icu_high_fio2, f_high_fio2 = sample_fio2_population(
        X_base,
        y,
        icu,
        fio2_values,
        fio2_min=0.50
    )

    # =========================
    # PLOT WEIGHT DISTRIBUTIONS
    # =========================

    plot_fio2_kde(
        f_all=fio2_values,
        f_reference=f_reference,
        f_high_fio2=f_high_fio2,
        out_dir="fio2_results"
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

    print("\nReference FiO2 cohort")
    print(f"N = {len(y_reference)}")
    print(f"Mean FiO2 = {np.mean(f_reference):.3f}")
    print(f"Range = {f_reference.min():.3f} - {f_reference.max():.3f}")
    print(f"Mortality = {np.mean(y_reference):.3f}")

    print("\nHigh-FiO2 cohort")
    print(f"N = {len(y_high_fio2)}")
    print(f"Mean FiO2 = {np.mean(f_high_fio2):.3f}")
    print(f"Range = {f_high_fio2.min():.3f} - {f_high_fio2.max():.3f}")
    print(f"Mortality = {np.mean(y_high_fio2):.3f}")
    os.makedirs("fio2_results", exist_ok=True)
    df_ml = run_all_models(
        models=models,

        X_reference=X_reference,
        y_reference=y_reference,
        icu_reference=icu_reference,

        X_high_fio2=X_high_fio2,
        y_high_fio2=y_high_fio2,
        icu_high_fio2=icu_high_fio2,

        out_dir="fio2_results"
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
        fio2_values=fio2_values,
        out_dir="fio2_results"
    )

    df_dl["Family"] = "Deep"

    df = pd.concat([df_ml, df_dl], ignore_index=True)
    final_df = df
    final_df.to_csv(
        "fio2_results/fio2_summary.csv",
        index=False
    )
    print("\nCombined Results:")
    print(final_df)

    if return_df:
        return final_df


if __name__ == "__main__":
    main()
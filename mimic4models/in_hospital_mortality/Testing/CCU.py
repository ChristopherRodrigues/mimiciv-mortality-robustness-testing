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
# CCU UNIT NAME
# =========================
CCU_UNIT = "Coronary Care Unit (CCU)"

# =========================
# UTILITIES
# =========================

def safe_float(x):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except:
        pass
    return np.nan

def clip_outliers(values, feature_name):
    ranges = {
        "systolic blood pressure":  (40,  300),
        "diastolic blood pressure": (20,  200),
        "heart rate":               (20,  250),
    }
    key = feature_name.lower()
    if key not in ranges:
        return values
    low, high = ranges[key]
    return values[(values >= low) & (values <= high)]

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
# FEATURE EXTRACTION
# =========================

def extract_feature_series(X_raw, feature_names, feature_query):
    feature_names = np.asarray(feature_names)
    idx = None
    for i, f in enumerate(feature_names):
        if feature_query.lower() in str(f).lower():
            idx = i
            break
    if idx is None:
        raise ValueError("Feature not found")
    values = []
    for patient in X_raw:
        for row in patient:
            try:
                v = safe_float(row[idx])
                if np.isfinite(v):
                    values.append(v)
            except:
                continue
    return np.array(values)

def extract_feature_series_by_icu(X_raw, feature_names, feature_query, icu, target="CCU"):
    feature_names = np.asarray(feature_names)
    idx = None
    for i, f in enumerate(feature_names):
        if feature_query.lower() in str(f).lower():
            idx = i
            break
    if idx is None:
        raise ValueError("Feature not found")
    values = []
    for patient, unit in zip(X_raw, icu):
        if target == "CCU":
            keep = str(unit) == CCU_UNIT
        elif target == "NON_CCU":
            keep = str(unit) != CCU_UNIT
        else:
            raise ValueError("target must be 'CCU' or 'NON_CCU'")
        if not keep:
            continue
        for row in patient:
            try:
                v = safe_float(row[idx])
                if np.isfinite(v):
                    values.append(v)
            except:
                continue
    return np.array(values)

def patient_level_delta(X1, X2, feature_names, feature_query):
    idx = np.where(
        np.char.find(
            np.char.lower(np.array(feature_names).astype(str)),
            feature_query.lower()
        ) >= 0
    )[0][0]
    deltas = []
    for p1, p2 in zip(X1, X2):
        v1, v2 = [], []
        for r1, r2 in zip(p1, p2):
            a = safe_float(r1[idx])
            b = safe_float(r2[idx])
            if np.isfinite(a) and np.isfinite(b):
                v1.append(a)
                v2.append(b)
        if len(v1):
            deltas.append(np.mean(np.array(v2) - np.array(v1)))
        else:
            deltas.append(np.nan)
    return np.array(deltas)

# =========================
# CCU PERTURBATION (RAW)
# =========================

def apply_ccu_perturbation_raw(X, icu, severity, feature_names):
    """
    Apply CCU cardiac haemodynamic perturbations to raw MIMIC time series.

    Perturbations:
        Minimal  — Mild hypertension / slightly reduced HR variability
        Moderate — Moderate hypertension / moderately reduced HR variability
        Severe   — Severe hypertension / markedly reduced HR variability
    """
    X = list(X)
    icu = np.asarray(icu)
    feature_names = np.asarray(feature_names)

    params = {
        "Minimal": {
            "sbp_shift":      5,
            "dbp_shift":      5,
            "hr_variability": 0.90,
        },
        "Moderate": {
            "sbp_shift":      10,
            "dbp_shift":      10,
            "hr_variability": 0.75,
        },
        "Severe": {
            "sbp_shift":      20,
            "dbp_shift":      20,
            "hr_variability": 0.50,
        },
    }[severity]

    def find_feature(names):
        for i, f in enumerate(feature_names):
            fl = str(f).lower()
            if any(name.lower() in fl for name in names):
                return i
        raise ValueError(f"Couldn't find {names}")

    sbp_i = find_feature(["Systolic blood pressure"])
    dbp_i = find_feature(["Diastolic blood pressure"])
    hr_i  = find_feature(["Heart Rate"])

    for patient, unit in zip(X, icu):
        if str(unit) != CCU_UNIT:
            continue

        # --------------------------------------------------
        # Blood pressure shifts
        # --------------------------------------------------
        for row in patient:

            # SBP — absolute shift
            if row[sbp_i] != "":
                v = safe_float(row[sbp_i])
                if np.isfinite(v) and 20 <= v <= 300:
                    row[sbp_i] = str(np.clip(v + params["sbp_shift"], 40, 300))

            # DBP — absolute shift
            if row[dbp_i] != "":
                v = safe_float(row[dbp_i])
                if np.isfinite(v) and 20 <= v <= 200:
                    row[dbp_i] = str(np.clip(v + params["dbp_shift"], 20, 200))

        # --------------------------------------------------
        # Reduce HR variability (preserve mean)
        # --------------------------------------------------
        hr_values, valid_rows = [], []
        for row in patient:
            v = safe_float(row[hr_i])
            if np.isfinite(v):
                hr_values.append(v)
                valid_rows.append(row)

        if len(hr_values) == 0:
            continue

        mean_hr = np.mean(hr_values)
        factor  = params["hr_variability"]
        for row in valid_rows:
            v      = safe_float(row[hr_i])
            hr_new = np.clip(mean_hr + (v - mean_hr) * factor, 20, 250)
            row[hr_i] = str(hr_new)

    return X

# =========================
# CCU PERTURBATION (TENSOR)
# =========================

def apply_ccu_perturbation_tensor(X, icu, severity, feature_names):
    """
    Apply CCU perturbation to denormalized deep-learning tensors.
    X must have shape (N, T, F).
    """
    X = X.copy()
    icu = np.asarray(icu)
    feature_names = np.asarray(feature_names)

    params = {
        "Minimal": {
            "sbp_shift":      5,
            "dbp_shift":      5,
            "hr_variability": 0.90,
        },
        "Moderate": {
            "sbp_shift":      10,
            "dbp_shift":      10,
            "hr_variability": 0.75,
        },
        "Severe": {
            "sbp_shift":      20,
            "dbp_shift":      20,
            "hr_variability": 0.50,
        },
    }[severity]

    def find_feature(names):
        for name in names:
            matches = np.where(
                np.char.find(
                    np.char.lower(feature_names.astype(str)),
                    name.lower()
                ) >= 0
            )[0]
            if len(matches):
                return matches[0]
        raise ValueError(f"Missing feature {names}")

    sbp_i = find_feature(["Systolic blood pressure"])
    dbp_i = find_feature(["Diastolic blood pressure"])
    hr_i  = find_feature(["Heart Rate"])

    ccu_mask = np.array([str(x) == CCU_UNIT for x in icu])
    if ccu_mask.sum() == 0:
        return X
    if X.ndim != 3:
        raise RuntimeError("CCU tensor perturbation only supports (N,T,F) tensors.")

    # --------------------------------------------------
    # Blood pressure shifts
    # --------------------------------------------------
    for feat_i, lo, hi, shift in [
        (sbp_i, 40, 300, params["sbp_shift"]),
        (dbp_i, 20, 200, params["dbp_shift"]),
    ]:
        bp = X[ccu_mask, :, feat_i]
        valid = bp > 0
        bp[valid] += shift
        X[ccu_mask, :, feat_i] = np.clip(bp, lo, hi)

    # --------------------------------------------------
    # Reduce HR variability (preserve mean)
    # --------------------------------------------------
    factor = params["hr_variability"]
    hr     = X[ccu_mask, :, hr_i]
    valid  = hr > 0
    count  = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    hr_mean = (hr * valid).sum(axis=1, keepdims=True) / count
    hr_new  = hr_mean + (hr - hr_mean) * factor
    hr      = np.where(valid, hr_new, hr)
    X[ccu_mask, :, hr_i] = np.clip(hr, 20, 250)

    return X

# =========================
# NORMALIZE / DENORMALIZE
# =========================

def inverse_normalize(X, normalizer, cont_channels):
    X = X.copy()
    means = np.array(normalizer._means)
    stds  = np.array(normalizer._stds)
    for ch in cont_channels:
        X[:, :, ch] = X[:, :, ch] * stds[ch] + means[ch]
    return X

def renormalize(X, normalizer, cont_channels):
    X = X.copy()
    means = np.array(normalizer._means)
    stds  = np.array(normalizer._stds)
    for ch in cont_channels:
        X[:, :, ch] = (X[:, :, ch] - means[ch]) / stds[ch]
    return X

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

def run_experiment(pipeline_path, X_base, X_pert, y, icu, severity="Moderate", out_dir="."):
    os.makedirs(out_dir, exist_ok=True)
    pipeline = load_pipeline(pipeline_path)
    model = pipeline["model"]

    def transform(X_in):
        X_in = X_in[:, pipeline["non_empty_cols"]]
        X_in = pipeline["imputer"].transform(X_in)
        X_in = pipeline["variance_selector"].transform(X_in)
        if len(pipeline["correlated_columns_removed"]) > 0:
            drop = set(pipeline["correlated_columns_removed"])
            mask = np.array([f not in drop for f in pipeline["feature_names_after_variance"]])
            X_in = X_in[:, mask]
        if "scaler" in pipeline and pipeline["scaler"] is not None:
            X_in = pipeline["scaler"].transform(X_in)
        icu_enc = pipeline["icu_encoder"].transform(np.array(icu).reshape(-1, 1))
        X_in = np.hstack([X_in, icu_enc])
        return X_in

    X_base_t = transform(X_base)
    X_pert_t = transform(X_pert)

    base_metrics = evaluate(model, X_base_t, y)
    pert_metrics = evaluate(model, X_pert_t, y)
    shift = np.mean(np.abs(get_probs(model, X_pert_t) - get_probs(model, X_base_t)))

    return {
        "baseline":         base_metrics,
        "perturbed":        pert_metrics,
        "prediction_shift": shift,
    }

# =========================
# MULTI-MODEL RUNNER (CLASSICAL)
# =========================

def run_all_models(models, X_base, X_pert, y, icu, feature_names, severity="Moderate", out_dir="."):
    results = []
    for name, path in models.items():
        print(f"Running {name}...")
        res = run_experiment(path, X_base, X_pert, y, icu, severity, out_dir)
        results.append({
            "Model":      name,
            "AUROC_drop": res["perturbed"]["AUROC"] - res["baseline"]["AUROC"],
            "AUPRC_drop": res["perturbed"]["AUPRC"] - res["baseline"]["AUPRC"],
            "Brier_drop": res["perturbed"]["Brier"] - res["baseline"]["Brier"],
            "ECE_drop":   res["perturbed"]["ECE"]   - res["baseline"]["ECE"],
            "MCE_drop":   res["perturbed"]["MCE"]   - res["baseline"]["MCE"],
            "Shift":      res["prediction_shift"],
        })
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "ccu_summary.csv"), index=False)
    return df

# =========================
# MULTI-MODEL RUNNER (DEEP)
# =========================

def run_all_models_deep(models: dict,
                        snapshot_path,
                        pipeline_map: dict,
                        severity="Moderate",
                        out_dir="."):
    os.makedirs(out_dir, exist_ok=True)
    results = []

    snapshot = joblib.load(snapshot_path)
    X_norm = snapshot["X"]
    y      = np.array(snapshot["y"])
    icu    = snapshot["icu"]

    for name, model_path in models.items():
        print(f"Running deep model: {name}...")

        pipeline = load_pipeline(pipeline_map[name])

        if name == "LSTM":
            from mimic4models.keras_models.lstm import Network
            model = Network(dim=16, batch_norm=False, dropout=0.3, rec_dropout=0.0,
                            task="ihm", target_repl=False, deep_supervision=False,
                            num_classes=1, depth=2, input_dim=96)
            model.load_weights(model_path)

        elif name == "ChannelwiseLSTM":
            from mimic4models.keras_models.channel_wise_lstms import Network
            model = Network(dim=16, batch_norm=False, dropout=0.3, rec_dropout=0.0,
                            header=pipeline["feature_names"], task="ihm",
                            target_repl=False, deep_supervision=False,
                            depth=2, input_dim=96, size_coef=4.0)
            model.load_weights(model_path)

        elif name == "LSTM_DS":
            from mimic4models.keras_models.lstm import Network
            model = Network(dim=16, batch_norm=False, dropout=0.3, rec_dropout=0.0,
                            task="ihm", target_repl=True, deep_supervision=False,
                            num_classes=1, depth=2, input_dim=96)
            model.load_weights(model_path)

        elif name == "ChannelwiseLSTM_DS":
            from mimic4models.keras_models.channel_wise_lstms import Network
            model = Network(dim=16, batch_norm=False, dropout=0.3, rec_dropout=0.0,
                            header=pipeline["feature_names"], task="ihm",
                            target_repl=True, deep_supervision=False,
                            depth=2, input_dim=96, size_coef=4.0)
            model.load_weights(model_path)

        feature_names = pipeline["feature_names"]
        normalizer    = pipeline["normalizer"]
        cont_channels = pipeline["cont_channels"]

        # 1. BASELINE
        X_base = X_norm.copy()

        # 2. DENORMALIZE
        X_base_raw = inverse_normalize(X_base, normalizer, cont_channels)

        # 3. PERTURB IN RAW SPACE
        X_pert_raw = apply_ccu_perturbation_tensor(
            X_base_raw.copy(), icu, severity, feature_names
        )

        # 4. RENORMALIZE
        X_pert = renormalize(X_pert_raw, normalizer, cont_channels)
        assert X_base.shape == X_pert.shape, "Shape mismatch after perturbation"
        assert not np.isnan(X_pert).any(),   "NaNs introduced in perturbation"

        # 5. PREDICT
        p_base = model.predict(X_base)
        if isinstance(p_base, list):
            p_base = p_base[0]
        p_base = np.asarray(p_base).ravel()

        p_pert = model.predict(X_pert)
        if isinstance(p_pert, list):
            p_pert = p_pert[0]
        p_pert = np.asarray(p_pert).ravel()

        # 6. METRICS
        base_metrics = evaluate_deep(p_base, y)
        pert_metrics = evaluate_deep(p_pert, y)
        shift = np.mean(np.abs(p_pert - p_base))

        results.append({
            "Model":      name,
            "AUROC_drop": pert_metrics["AUROC"] - base_metrics["AUROC"],
            "AUPRC_drop": pert_metrics["AUPRC"] - base_metrics["AUPRC"],
            "Brier_drop": pert_metrics["Brier"] - base_metrics["Brier"],
            "ECE_drop":   pert_metrics["ECE"]   - base_metrics["ECE"],
            "MCE_drop":   pert_metrics["MCE"]   - base_metrics["MCE"],
            "Shift":      shift,
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "ccu_deep_summary.csv"), index=False)
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
    parser.add_argument('--severity', type=str, default='Moderate',
                        choices=['Minimal', 'Moderate', 'Severe'])
    args = parser.parse_args(cli_args)
    print(args)

    severity = args.severity

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
    y             = test_ret['y']
    icu           = test_icu
    feature_names = test_ret['header']

    # BASELINE features (classical)
    X_base = common_utils.extract_features_from_rawdata(
        X_raw, test_ret['header'], args.period, args.features
    )

    # PERTURB raw
    X_raw_pert = apply_ccu_perturbation_raw(
        copy.deepcopy(X_raw), icu, severity, test_ret['header']
    )

    # Sanity checks
    icu_arr  = np.array(icu)
    ccu_mask = np.array([str(x) == CCU_UNIT for x in icu_arr])
    print("CCU patients:", ccu_mask.sum())
    print("Total patients:", len(ccu_mask))
    print("Fraction:", ccu_mask.mean())

    for feat_query in ["systolic blood pressure", "diastolic blood pressure", "heart rate"]:
        try:
            delta = patient_level_delta(X_raw, X_raw_pert, feature_names, feat_query)
            print(f"Mean {feat_query} shift: {np.nanmean(delta):.4f}")
            print(f"Max  {feat_query} shift: {np.nanmax(np.abs(delta)):.4f}")
        except Exception as e:
            print(f"Could not compute delta for {feat_query}: {e}")

    # KDE plots — CCU only and Non-CCU control
    plot_features = [
        ("Systolic BP",  "systolic blood pressure"),
        ("Diastolic BP", "diastolic blood pressure"),
        ("Heart Rate",   "heart rate"),
    ]

    result_dir = r"C:/Users/chris/Thesis/mimic4-benchmarks/ccu_results/"
    os.makedirs(result_dir, exist_ok=True)

    for label, target in [("CCU", "CCU"), ("Non-CCU control", "NON_CCU")]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes = axes.flatten()

        for i, (feat_label, feat_query) in enumerate(plot_features):
            try:
                base = extract_feature_series_by_icu(
                    X_raw, feature_names, feat_query, icu_arr, target=target)
                pert = extract_feature_series_by_icu(
                    X_raw_pert, feature_names, feat_query, icu_arr, target=target)
                base = clip_outliers(base[np.isfinite(base)], feat_query)
                pert = clip_outliers(pert[np.isfinite(pert)], feat_query)
                sns.kdeplot(base, ax=axes[i], label=f"{label} Baseline", common_norm=False)
                sns.kdeplot(pert, ax=axes[i], label=f"{label} Perturbed", common_norm=False)
            except Exception as e:
                axes[i].set_title(f"{feat_label} — N/A")
                print(f"KDE skipped for {feat_label}: {e}")
                continue
            axes[i].set_title(f"{feat_label} ({label})")
            axes[i].set_xlabel("")
            axes[i].set_ylabel("Density")
            axes[i].legend()

        plt.suptitle(f"CCU Perturbation KDEs — {severity} — {label}", fontsize=14)
        plt.tight_layout()
        fname = f"ccu_kde_{severity.lower()}_{target.lower()}.png"
        plt.savefig(os.path.join(result_dir, fname))
        plt.close()

    # Perturbed features for classical models
    X_pert = common_utils.extract_features_from_rawdata(
        X_raw_pert, test_ret['header'], args.period, args.features
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

    sev = args.severity
    print(f"\n===== {sev} =====")

    df_ml = run_all_models(
        models=models,
        X_base=X_base,
        X_pert=X_pert,
        y=y,
        icu=icu,
        feature_names=feature_names,
        severity=sev,
        out_dir=f"ccu_results/{sev}"
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
        severity=sev,
        out_dir=f"ccu_results/{sev}"
    )
    df_dl["Family"] = "Deep"

    df = pd.concat([df_ml, df_dl], ignore_index=True)
    df["Severity"] = sev
    final_df = df

    final_df.to_csv(f"ccu_results/{sev}_summary.csv", index=False)
    print("\nCombined Results:")
    print(final_df)

    if return_df:
        return final_df


if __name__ == "__main__":
    main()
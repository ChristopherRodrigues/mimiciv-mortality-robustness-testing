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
# SICU UNIT NAME
# =========================
SICU_UNIT = "Surgical Intensive Care Unit (SICU)"

# =========================
# METRICS
# =========================

def read_and_extract_features(reader, period, features, icu_map):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    y = ret['y']
    names = ret['name']
    icu = [icu_map.get(n, "UNKNOWN") for n in names]
    return (X, y, names, ret['header'], icu)

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

def extract_feature_series_by_icu(X_raw, feature_names, feature_query, icu, target="SICU"):
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
        if target == "SICU":
            keep = str(unit) == SICU_UNIT
        elif target == "NON_SICU":
            keep = str(unit) != SICU_UNIT
        else:
            raise ValueError("target must be 'SICU' or 'NON_SICU'")
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
        "heart rate":               (20,  250),
        "temperature":              (25,   45),
        "respiratory rate":         ( 4,   60),
        "hemoglobin":               ( 2,   25),
        "hematocrit":               ( 5,   65),
        "lactate":                  (0.1,  30),
        "systolic blood pressure":  (40,  300),
        "diastolic blood pressure": (20,  200),
        "white blood cell count":   (0,  100),
        "fraction inspired oxygen": (0.21,  1.00),
        "creatinine":               (0.1,  20),
    }
    key = feature_name.lower()
    if key not in ranges:
        return values
    low, high = ranges[key]
    return values[(values >= low) & (values <= high)]

# =========================
# SICU PERTURBATION (RAW)
# =========================
def apply_sicu_perturbation_raw(X, icu, severity, feature_names):
    """
    Apply SICU post-operative inflammation perturbations to raw MIMIC time series.

    Perturbations:
        Minimal  — Normal post-operative inflammation
        Moderate — Complicated inflammatory response
        Severe   — Severe post-operative systemic inflammation / sepsis-like
    """
    X = list(X)
    icu = np.asarray(icu)
    feature_names = np.asarray(feature_names)

    params = {
        "Minimal": {
            "temp_shift":   0.3,
            "hr_pct":       0.05,
            "rr_pct":       0.05,
            "wbc_pct":      0.15,
            "lactate_pct":  0.0,
            "hgb_pct":     -0.05,
            "hct_pct":     -0.05,
            "fio2_pct":     0.0,
            "creat_pct":    0.0,
        },
        "Moderate": {
            "temp_shift":   0.8,
            "hr_pct":       0.10,
            "rr_pct":       0.10,
            "wbc_pct":      0.30,
            "lactate_pct":  0.25,
            "hgb_pct":     -0.10,
            "hct_pct":     -0.10,
            "fio2_pct":     0.10,
            "creat_pct":    0.0,
        },
        "Severe": {
            "temp_shift":   1.5,
            "hr_pct":       0.20,
            "rr_pct":       0.20,
            "wbc_pct":      0.50,
            "lactate_pct":  0.50,
            "hgb_pct":     -0.20,
            "hct_pct":     -0.20,
            "fio2_pct":     0.20,
            "creat_pct":    0.20,
        },
    }[severity]

    def find_feature(names):
        for name in names:
            for i, f in enumerate(feature_names):
                if name.lower() in str(f).lower():
                    return i
        raise ValueError(f"Couldn't find {names}")

    hr_i   = find_feature(["Heart Rate"])
    temp_i = find_feature(["Temperature"])
    rr_i   = find_feature(["Respiratory rate"])
    hgb_i  = find_feature(["Hemoglobin"])
    hct_i  = find_feature(["Hematocrit"])

    try:
        wbc_i = find_feature(["White blood cell count"])
        has_wbc = True
    except ValueError:
        has_wbc = False
        print("Warning: WBC feature not found, skipping WBC perturbation.")

    try:
        lac_i = find_feature(["Lactate"])
        has_lactate = True
    except ValueError:
        has_lactate = False
        print("Warning: Lactate feature not found, skipping lactate perturbation.")

    try:
        fio2_i = find_feature(["fraction inspired oxygen"])
        has_fio2 = True
    except ValueError:
        has_fio2 = False
        print("Warning: FiO2 feature not found, skipping FiO2 perturbation.")

    try:
        creat_i = find_feature(["Creatinine"])
        has_creat = True
    except ValueError:
        has_creat = False
        print("Warning: Creatinine feature not found, skipping creatinine perturbation.")

    for patient, unit in zip(X, icu):
        if str(unit) != SICU_UNIT:
            continue

        for row in patient:

            # Temperature — absolute shift (°C)
            if row[temp_i] != "":
                v = safe_float(row[temp_i])
                if np.isfinite(v) and 25 <= v <= 45:
                    row[temp_i] = str(np.clip(v + params["temp_shift"], 25, 45))

            # Heart Rate — percentage increase
            if row[hr_i] != "":
                v = safe_float(row[hr_i])
                if np.isfinite(v) and 20 <= v <= 250:
                    row[hr_i] = str(np.clip(v * (1 + params["hr_pct"]), 20, 250))

            # Respiratory Rate — percentage increase
            if row[rr_i] != "":
                v = safe_float(row[rr_i])
                if np.isfinite(v) and 4 <= v <= 60:
                    row[rr_i] = str(np.clip(v * (1 + params["rr_pct"]), 4, 60))

            # WBC — percentage increase
            if has_wbc and params["wbc_pct"] > 0 and row[wbc_i] != "":
                v = safe_float(row[wbc_i])
                if np.isfinite(v) and 0 <= v <= 100:
                    row[wbc_i] = str(np.clip(v * (1 + params["wbc_pct"]), 0, 100))

            # Lactate — percentage increase
            if has_lactate and params["lactate_pct"] > 0 and row[lac_i] != "":
                v = safe_float(row[lac_i])
                if np.isfinite(v) and 0.1 <= v <= 30:
                    row[lac_i] = str(np.clip(v * (1 + params["lactate_pct"]), 0.1, 30))

            # Hemoglobin — percentage decrease
            if row[hgb_i] != "":
                v = safe_float(row[hgb_i])
                if np.isfinite(v) and 2 <= v <= 25:
                    row[hgb_i] = str(np.clip(v * (1 + params["hgb_pct"]), 2, 25))

            # Hematocrit — percentage decrease
            if row[hct_i] != "":
                v = safe_float(row[hct_i])
                if np.isfinite(v) and 5 <= v <= 65:
                    row[hct_i] = str(np.clip(v * (1 + params["hct_pct"]), 5, 65))

            # FiO2 — percentage increase 
            if has_fio2 and params["fio2_pct"] > 0 and row[fio2_i] != "":
                v = safe_float(row[fio2_i])
                if np.isfinite(v) and 0.21 <= v <= 1:
                    row[fio2_i] = str(np.clip(v * (1 + params["fio2_pct"]), 0.21, 1.00))

            # Creatinine — percentage increase
            if has_creat and params["creat_pct"] > 0 and row[creat_i] != "":
                v = safe_float(row[creat_i])
                if np.isfinite(v) and 0.1 <= v <= 20:
                    row[creat_i] = str(np.clip(v * (1 + params["creat_pct"]), 0.1, 20))

    return X

# =========================
# SICU PERTURBATION (TENSOR)
# =========================
def apply_sicu_perturbation_tensor(X, icu, severity, feature_names):
    """
    Apply SICU perturbation to denormalized deep-learning tensors.
    X must have shape (N, T, F).
    """
    X = X.copy()
    icu = np.asarray(icu)
    feature_names = np.asarray(feature_names)

    params = {
        "Minimal": {
            "temp_shift":   0.3,
            "hr_pct":       0.05,
            "rr_pct":       0.05,
            "wbc_pct":      0.15,
            "lactate_pct":  0.0,
            "hgb_pct":     -0.05,
            "hct_pct":     -0.05,
            "fio2_pct":     0.0,
            "creat_pct":    0.0,
        },
        "Moderate": {
            "temp_shift":   0.8,
            "hr_pct":       0.10,
            "rr_pct":       0.10,
            "wbc_pct":      0.30,
            "lactate_pct":  0.25,
            "hgb_pct":     -0.10,
            "hct_pct":     -0.10,
            "fio2_pct":     0.10,
            "creat_pct":    0.0,
        },
        "Severe": {
            "temp_shift":   1.5,
            "hr_pct":       0.20,
            "rr_pct":       0.20,
            "wbc_pct":      0.50,
            "lactate_pct":  0.50,
            "hgb_pct":     -0.20,
            "hct_pct":     -0.20,
            "fio2_pct":     0.20,
            "creat_pct":    0.20,
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

    hr_i   = find_feature(["Heart Rate"])
    temp_i = find_feature(["Temperature"])
    rr_i   = find_feature(["Respiratory rate"])
    hgb_i  = find_feature(["Hemoglobin"])
    hct_i  = find_feature(["Hematocrit"])

    try:
        wbc_i = find_feature(["White blood cell count"])
        has_wbc = True
    except ValueError:
        has_wbc = False

    try:
        lac_i = find_feature(["Lactate"])
        has_lactate = True
    except ValueError:
        has_lactate = False

    try:
        fio2_i = find_feature(["fraction inspired oxygen"])
        has_fio2 = True
    except ValueError:
        has_fio2 = False

    try:
        creat_i = find_feature(["Creatinine"])
        has_creat = True
    except ValueError:
        has_creat = False

    sicu_mask = np.array([str(x) == SICU_UNIT for x in icu])
    if sicu_mask.sum() == 0:
        return X
    if X.ndim != 3:
        raise RuntimeError("SICU tensor perturbation only supports (N,T,F) tensors.")

    def pct_change(arr, pct, lo, hi):
        mask = arr > 0
        arr[mask] = np.clip(arr[mask] * (1 + pct), lo, hi)
        return arr

    # Temperature — absolute shift
    temp = X[sicu_mask, :, temp_i]
    mask = temp > 0
    temp[mask] = np.clip(temp[mask] + params["temp_shift"], 25, 45)
    X[sicu_mask, :, temp_i] = temp

    # Heart Rate
    X[sicu_mask, :, hr_i] = pct_change(X[sicu_mask, :, hr_i], params["hr_pct"], 20, 250)

    # Respiratory Rate
    X[sicu_mask, :, rr_i] = pct_change(X[sicu_mask, :, rr_i], params["rr_pct"], 4, 60)

    # WBC
    if has_wbc and params["wbc_pct"] > 0:
        X[sicu_mask, :, wbc_i] = pct_change(X[sicu_mask, :, wbc_i], params["wbc_pct"], 0, 100)

    # Lactate
    if has_lactate and params["lactate_pct"] > 0:
        X[sicu_mask, :, lac_i] = pct_change(X[sicu_mask, :, lac_i], params["lactate_pct"], 0.1, 30)

    # Hemoglobin
    X[sicu_mask, :, hgb_i] = pct_change(X[sicu_mask, :, hgb_i], params["hgb_pct"], 2, 25)

    # Hematocrit
    X[sicu_mask, :, hct_i] = pct_change(X[sicu_mask, :, hct_i], params["hct_pct"], 5, 65)

    # FiO2
    if has_fio2 and params["fio2_pct"] > 0:
        X[sicu_mask, :, fio2_i] = pct_change(X[sicu_mask, :, fio2_i], params["fio2_pct"], 0.21, 1.00)

    # Creatinine
    if has_creat and params["creat_pct"] > 0:
        X[sicu_mask, :, creat_i] = pct_change(X[sicu_mask, :, creat_i], params["creat_pct"], 0.1, 20)

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
        "ECE": ece,
        "MCE": mce
    }

def evaluate_deep(y_prob, y):
    y = np.asarray(y)
    ece, mce = expected_calibration_error(y, y_prob)
    return {
        "AUROC": roc_auc_score(y, y_prob),
        "AUPRC": average_precision_score(y, y_prob),
        "Brier": brier_score_loss(y, y_prob),
        "ECE": ece,
        "MCE": mce
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
        "baseline": base_metrics,
        "perturbed": pert_metrics,
        "prediction_shift": shift
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
            "Model": name,
            "AUROC_drop": res["perturbed"]["AUROC"] - res["baseline"]["AUROC"],
            "AUPRC_drop": res["perturbed"]["AUPRC"] - res["baseline"]["AUPRC"],
            "Brier_drop": res["perturbed"]["Brier"] - res["baseline"]["Brier"],
            "ECE_drop":   res["perturbed"]["ECE"]   - res["baseline"]["ECE"],
            "MCE_drop":   res["perturbed"]["MCE"]   - res["baseline"]["MCE"],
            "Shift": res["prediction_shift"]
        })
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "sicu_summary.csv"), index=False)
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

        feature_names  = pipeline["feature_names"]
        normalizer     = pipeline["normalizer"]
        cont_channels  = pipeline["cont_channels"]

        # 1. BASELINE
        X_base = X_norm.copy()

        # 2. DENORMALIZE
        X_base_raw = inverse_normalize(X_base, normalizer, cont_channels)

        # 3. PERTURB IN RAW SPACE
        X_pert_raw = apply_sicu_perturbation_tensor(
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
            "Shift": shift
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "sicu_deep_summary.csv"), index=False)
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
    X_raw_pert = apply_sicu_perturbation_raw(
        copy.deepcopy(X_raw), icu, severity, test_ret['header']
    )

    # Sanity checks
    icu_arr    = np.array(icu)
    sicu_mask  = np.array([str(x) == SICU_UNIT for x in icu_arr])
    print("SICU patients:", sicu_mask.sum())
    print("Total patients:", len(sicu_mask))
    print("Fraction:", sicu_mask.mean())

    for feat_query in ["heart rate", "temperature", "hemoglobin", "hematocrit",
                       "lactate", "white blood cell count", "fraction inspired oxygen", "creatinine"]:
        try:
            delta = patient_level_delta(X_raw, X_raw_pert, feature_names, feat_query)
            print(f"Mean {feat_query} shift: {np.nanmean(delta):.4f}")
            print(f"Max  {feat_query} shift: {np.nanmax(np.abs(delta)):.4f}")
        except Exception as e:
            print(f"Could not compute delta for {feat_query}: {e}")

    # KDE plots — SICU only vs Non-SICU control
    plot_features = [
        ("Heart Rate",       "heart rate"),
        ("Temperature",      "temperature"),
        ("Respiratory rate", "respiratory rate"),
        ("Hemoglobin",       "hemoglobin"),
        ("Hematocrit",       "hematocrit"),
        ("Lactate",          "lactate"),
        ("WBC",              "white blood cell count"),
        ("FiO2",             "fraction inspired oxygen"),
        ("Creatinine",       "creatinine"),
    ]

    result_dir = r"C:/Users/chris/Thesis/mimic4-benchmarks/sicu_results/"
    os.makedirs(result_dir, exist_ok=True)

    for label, target in [("SICU", "SICU"), ("Non-SICU control", "NON_SICU")]:
        n_feats = len(plot_features)
        ncols = 3
        nrows = (n_feats + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4))
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

        # Hide any unused subplots
        for j in range(n_feats, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(f"SICU Perturbation KDEs — {severity} — {label}", fontsize=14)
        plt.tight_layout()
        fname = f"sicu_kde_{severity.lower()}_{target.lower()}.png"
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
        out_dir=f"sicu_results/{sev}"
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
        out_dir=f"sicu_results/{sev}"
    )
    df_dl["Family"] = "Deep"

    df = pd.concat([df_ml, df_dl], ignore_index=True)
    df["Severity"] = sev
    final_df = df

    final_df.to_csv(f"sicu_results/{sev}_summary.csv", index=False)
    print("\nCombined Results:")
    print(final_df)

    if return_df:
        return final_df


if __name__ == "__main__":
    main()
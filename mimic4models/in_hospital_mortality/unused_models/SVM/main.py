from __future__ import absolute_import
from __future__ import print_function

print("Loading modules...", flush=True)
from mimic4benchmark.readers import InHospitalMortalityReader
from mimic4models import common_utils
from mimic4models.metrics import print_metrics_binary
from mimic4models.in_hospital_mortality.utils import save_results
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.impute import SimpleImputer
from mimic4models.feature_extractor import build_feature_names
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
from sklearn.calibration import calibration_curve
from mimic4models.metrics import plot_roc_curve, plot_pr_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve


import os
import numpy as np
import argparse
import json
import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import re

print("Modules loaded...", flush=True)
def read_and_extract_features(reader, period, features, icu_map):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    # ret = common_utils.read_chunk(reader, 100)
    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    y = ret['y']
    names = ret['name']

    icu = [icu_map.get(n, "UNKNOWN") for n in names]

    return (X, y, names, ret['header'], icu)
    #return (X, ret['y'], ret['name'], ret['header'])

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

            # sort to ensure consistent ordering
            stays_df = stays_df.sort_values(by="intime") if "intime" in stays_df.columns else stays_df

            icu_units = stays_df["LAST_CAREUNIT"].tolist()

            # map episodes → ICU units
            for i, icu in enumerate(icu_units):
                episode_idx = i + 1
                key = f"{subject_id}_episode{episode_idx}_timeseries.csv"
                icu_map[key] = icu

    return icu_map

def main():
    print("Starting Execution...", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', type=str, default='all', help='specifies which period extract features from',
                        choices=['first4days', 'first8days', 'last12hours', 'first25percent', 'first50percent', 'all'])
    parser.add_argument('--features', type=str, default='all', help='specifies what features to extract',
                        choices=['all', 'len', 'all_but_len'])
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                        default=os.path.join(os.path.dirname(__file__), "..\\..\\..\\data\\in-hospital-mortality\\"))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')
    args = parser.parse_args()
    print(args)

    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                             listfile=os.path.join(args.data, 'train_listfile.csv'),
                                             period_length=48.0)

    val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                           listfile=os.path.join(args.data, 'val_listfile.csv'),
                                           period_length=48.0)

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=48.0)
    icu_df = pd.read_csv(os.path.join(args.data, "../root/all_stays.csv"))
    icu_map = build_icu_map(os.path.join(args.data, "../root"))
    print('Reading data and extracting features ...', flush=True)
    (train_X, train_y, train_names, train_feature_header, train_icu) = read_and_extract_features(train_reader, args.period, args.features, icu_map)
    (val_X, val_y, val_names, val_feature_header, val_icu) = read_and_extract_features(val_reader, args.period, args.features, icu_map)
    (test_X, test_y, test_names, test_feature_header, test_icu) = read_and_extract_features(test_reader, args.period, args.features, icu_map)
    print('  train data shape = {}'.format(train_X.shape))
    print('  validation data shape = {}'.format(val_X.shape))
    print('  test data shape = {}'.format(test_X.shape))
    print("\nICU distribution (train):", flush=True)
    print(Counter(train_icu))

    print("\nICU distribution (val):", flush=True)
    print(Counter(val_icu))

    print("\nICU distribution (test):", flush=True)
    print(Counter(test_icu))

    print("\nSample names vs ICU mapping:", flush=True)
    for i in range(10):
        print(train_names[i], "->", train_icu[i])

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(np.array(train_icu).reshape(-1, 1))
    train_icu_enc = encoder.transform(np.array(train_icu).reshape(-1, 1))
    val_icu_enc   = encoder.transform(np.array(val_icu).reshape(-1, 1))
    test_icu_enc  = encoder.transform(np.array(test_icu).reshape(-1, 1))

    # Remove columns that are entirely NaN (based on training set ONLY)
    non_empty_cols = ~np.all(np.isnan(train_X), axis=0)

    train_X = train_X[:, non_empty_cols]
    val_X = val_X[:, non_empty_cols]
    test_X = test_X[:, non_empty_cols]
    print('Imputing missing values ...', flush=True)
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(train_X)
    train_X = np.array(imputer.transform(train_X), dtype=np.float32)
    val_X = np.array(imputer.transform(val_X), dtype=np.float32)
    test_X = np.array(imputer.transform(test_X), dtype=np.float32)

    # ---------------------------------------------------
    # Build feature names
    # ---------------------------------------------------

    feature_names = np.array(build_feature_names(train_feature_header))

    # remove columns that were entirely NaN
    feature_names = feature_names[non_empty_cols]

    # ---------------------------------------------------
    # Remove nearly constant features
    # ---------------------------------------------------

    print("\nRemoving low-variance features...")
    print("Features before VarianceThreshold:", train_X.shape[1])

    selector = VarianceThreshold(threshold=0.01)

    train_X = selector.fit_transform(train_X)
    val_X = selector.transform(val_X)
    test_X = selector.transform(test_X)

    # update feature names after variance filtering
    feature_names = feature_names[selector.get_support()]

    print("Features after VarianceThreshold:", train_X.shape[1])

    # ---------------------------------------------------
    # Convert to DataFrames for correlation filtering
    # ---------------------------------------------------

    train_X_df = pd.DataFrame(train_X, columns=feature_names)
    val_X_df = pd.DataFrame(val_X, columns=feature_names)
    test_X_df = pd.DataFrame(test_X, columns=feature_names)

    # ---------------------------------------------------
    # Remove highly correlated features
    # ---------------------------------------------------

    print("\nRemoving highly correlated features...")

    corr_matrix = train_X_df.corr().abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        column for column in upper.columns
        if any(upper[column] > 0.90)
    ]

    print(f"Removing {len(to_drop)} correlated features")

    train_X_df = train_X_df.drop(columns=to_drop)
    val_X_df = val_X_df.drop(columns=to_drop)
    test_X_df = test_X_df.drop(columns=to_drop)

    print("Remaining features:", train_X_df.shape[1])

    # final feature names after correlation filtering
    feature_names = train_X_df.columns.to_numpy()

    # ---------------------------------------------------
    # Convert back to numpy
    # ---------------------------------------------------

    train_X_reduced = train_X_df.values
    val_X_reduced = val_X_df.values
    test_X_reduced = test_X_df.values

    print('SVM needs normalization ...', flush=True)

    # -----------------------------
    # Scaling
    # -----------------------------
    scaler = StandardScaler()
    scaler.fit(train_X_reduced)

    train_X_scaled = scaler.transform(train_X_reduced)
    val_X_scaled   = scaler.transform(val_X_reduced)
    test_X_scaled  = scaler.transform(test_X_reduced)

    train_X_scaled = np.hstack([train_X_scaled, train_icu_enc])
    val_X_scaled   = np.hstack([val_X_scaled, val_icu_enc])
    test_X_scaled  = np.hstack([test_X_scaled, test_icu_enc])
    
    file_name = '{}.{}.svm'.format(args.period, args.features)

    svm = SVC(class_weight="balanced", random_state=42, probability=True)
    
    # Hyperparameter search space
    param_dist = {
    "C": loguniform(1e-2, 10),
    "gamma": loguniform(1e-4, 1e-1),
    "kernel": ["rbf"]
    }

    # Random search
    random_search = RandomizedSearchCV(
    estimator=svm,
    param_distributions=param_dist,
    n_iter=3,
    cv=3,
    scoring="roc_auc",
    verbose=2,
    random_state=42,
    n_jobs=-1
    )
    print("Starting SVM search...", flush=True)
    random_search.fit(train_X_scaled, train_y)
    print("Completed SVM search...", flush=True)
    print("Best parameters:", random_search.best_params_)
    print("Best score:", random_search.best_score_)

    # Train best model
    best_svm = random_search.best_estimator_

    best_svm.fit(train_X_scaled, train_y)

    # Calibration (VERY important for SVM)
    calibrated_svm = CalibratedClassifierCV(
    best_svm,
    method="sigmoid",   # much faster than isotonic
    cv=3)
    calibrated_svm.fit(train_X_scaled, train_y)

    joblib.dump(calibrated_svm, os.path.join(args.output_dir, "svm_model.pkl"))

    result_dir = os.path.join(args.output_dir, 'results')
    common_utils.create_directory(result_dir)

    # Predictions
    prediction = calibrated_svm.predict_proba(test_X_scaled)[:, 1]

    y_prob = prediction
    y_true = np.array(test_y)

    # ---------------------------------------------------
    # Threshold optimization using validation set
    # ---------------------------------------------------

    val_probs = calibrated_svm.predict_proba(val_X_scaled)[:, 1]

    precision, recall, thresholds = precision_recall_curve(val_y, val_probs)

    # avoid divide-by-zero
    f1_scores = 2 * precision[:-1] * recall[:-1] / (
       precision[:-1] + recall[:-1] + 1e-8
    )

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print("\nBest threshold from validation set:", best_threshold)
    print("Best validation F1:", f1_scores[best_idx])


    # -----------------------------
    # Calibration curve (overall)
    # -----------------------------
    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=10,
        strategy="quantile"
    )

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label="SVM")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Curve - Test Set")
    plt.legend()
    plt.savefig(os.path.join(result_dir, "svm_calibration_curve_test.png"), bbox_inches="tight")
    plt.close()
    plt.figure()
    plt.hist(y_prob, bins=50)
    plt.title("Distribution of predicted probabilities")
    plt.savefig(os.path.join(result_dir, "svm_prob_distribution.png"))
    plt.close()

    
    test_icu = np.array(test_icu)
    unique_icus = np.unique(test_icu)

    plt.figure(figsize=(8, 6))

    for icu in unique_icus:
        idx = (test_icu == icu)

        # skip very small groups (avoids noisy curves)
        if np.sum(idx) < 100:
            continue

        prob_true, prob_pred = calibration_curve(
            y_true[idx],
            prediction[idx],
            n_bins=10,
            strategy="quantile"
        )

        if len(prob_true) > 1:
            plt.plot(prob_pred, prob_true, marker="o", linewidth=1.5, label=icu)

    # perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect")

    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Curves by ICU (Test Set)")

    # Clean legend (important: too many ICUs otherwise messy)
    plt.legend(fontsize=8, loc="best", ncol=2)

    plt.grid(alpha=0.3)

    plt.savefig(os.path.join(result_dir, "svm_calibration_all_ICUs.png"), bbox_inches="tight")
    plt.close()
    # -----------------------------
    # ROC + PR curves
    # -----------------------------
    plot_roc_curve(y_true, y_prob, "SVM AUROC - Test Set", result_dir)
    plot_pr_curve(y_true, y_prob, "SVM AUPRC - Test Set", result_dir)

    icu_names = encoder.get_feature_names_out(["ICU"])
    feature_names = np.concatenate([feature_names, icu_names])

    print("train_X:", train_X.shape)
    print("val_X:", val_X.shape)
    print("test_X:", test_X.shape)

    # -----------------------------
    # SHAP 
    # -----------------------------
    background = shap.sample(train_X_scaled, 100, random_state=42)
    explainer = shap.Explainer(
    best_svm.predict_proba,
    masker=background,
    algorithm="permutation",
    max_evals=725
    )

    # sample for speed (important for MIMIC size)
    sample_idx = np.random.choice(len(test_X_scaled), min(100, len(test_X_scaled)), replace=False)
    X_sample = test_X_scaled[sample_idx]

    shap_values = explainer(X_sample)
    print(np.array(shap_values).shape)
    # binary classification fix
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    # fix final consistency check
    assert X_sample.shape[1] == len(feature_names)
    print("X shape:", X_sample.shape)
    print("features:", len(feature_names))
    # summary plot (beeswarm)
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False
    )
    plt.savefig(os.path.join(result_dir, "svm_test_shap_summary.png"), bbox_inches="tight")
    plt.close()

    # bar plot
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.savefig(os.path.join(result_dir, "svm_test_shap_bar.png"), bbox_inches="tight")
    plt.close()

    # Compute SHAP values for train split

    sample_idx_train = np.random.choice(len(train_X_scaled), 100, replace=False)
    X_sample_train = train_X_scaled[sample_idx_train]

    shap_values_train = explainer(X_sample_train)

    if isinstance(shap_values_train, list):
        shap_values_train = shap_values_train[1]
    elif shap_values_train.ndim == 3:
        shap_values_train = shap_values_train[:, :, 1]

    shap.summary_plot(shap_values_train, X_sample_train, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(result_dir, "svm_train_shap_summary.png"), bbox_inches='tight')
    plt.close()
    shap.summary_plot(shap_values_train, X_sample_train, feature_names=feature_names, plot_type="bar", show=False)
    plt.savefig(os.path.join(result_dir, "svm_train_shap_bar.png"), bbox_inches='tight')
    plt.close()

    # Compute SHAP values for val split
    sample_idx_val = np.random.choice(len(val_X_scaled), 100, replace=False)
    X_sample_val = val_X_scaled[sample_idx_val]
    shap_values_val = explainer(X_sample_val)

    if isinstance(shap_values_val, list):
        shap_values_val = shap_values_val[1]
    elif shap_values_val.ndim == 3:
        shap_values_val = shap_values_val[:, :, 1]
    shap.summary_plot(shap_values_val, X_sample_val, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(result_dir, "svm_val_shap_summary.png"), bbox_inches='tight')
    plt.close()
    shap.summary_plot(shap_values_val, X_sample_val, feature_names=feature_names, plot_type="bar", show=False)
    plt.savefig(os.path.join(result_dir, "svm_val_shap_bar.png"), bbox_inches='tight')
    plt.close()

    # -----------------------------
    # Metrics
    # -----------------------------
    train_metrics = print_metrics_binary(train_y, calibrated_svm.predict_proba(train_X_scaled), "Train Confusion Matrix", result_dir, threshold=best_threshold)
    val_metrics   = print_metrics_binary(val_y, calibrated_svm.predict_proba(val_X_scaled), "Validation Confusion Matrix", result_dir, threshold=best_threshold)
    test_metrics  = print_metrics_binary(test_y, calibrated_svm.predict_proba(test_X_scaled), "Test Confusion Matrix", result_dir, threshold=best_threshold)

    train_metrics = {k: float(v) for k, v in train_metrics.items()}
    val_metrics   = {k: float(v) for k, v in val_metrics.items()}
    test_metrics  = {k: float(v) for k, v in test_metrics.items()}

    json.dump(train_metrics, open(os.path.join(result_dir, f"train_{file_name}.json"), "w"))
    json.dump(val_metrics,   open(os.path.join(result_dir, f"val_{file_name}.json"), "w"))
    json.dump(test_metrics,  open(os.path.join(result_dir, f"test_{file_name}.json"), "w"))

    results_df = pd.DataFrame.from_dict({
    "Train": train_metrics,
    "Validation": val_metrics,
    "Test": test_metrics
    }, orient="index")

    results_df = results_df.rename(columns={
    "acc": "Accuracy",
    "prec0": "Precision (Class 0)",
    "prec1": "Precision (Class 1)",
    "rec0": "Recall (Class 0)",
    "rec1": "Recall (Class 1)",
    "auroc": "ROC-AUC",
    "auprc": "PR-AUC",
    "minpse": "min(P,Se)"
    })

    results_df = results_df.round(4)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print("\nModel Performance Summary:\n")
    print(results_df)

    save_results(test_names, prediction, test_y,
                 os.path.join(args.output_dir, 'predictions', file_name + '.csv'))

if __name__ == '__main__':
    print("Entering main: ", flush=True)
    main()

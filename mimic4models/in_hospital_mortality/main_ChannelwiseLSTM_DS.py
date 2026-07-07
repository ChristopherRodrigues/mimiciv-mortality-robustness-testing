from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import joblib
import os
import imp
import re
import tensorflow as tf
import json
import pandas as pd#
import seaborn as sns
import sys

from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import brier_score_loss

import shap
import matplotlib.pyplot as plt

from mimic4models.in_hospital_mortality import utils
from mimic4benchmark.readers import InHospitalMortalityReader

from mimic4models.preprocessing import Discretizer, Normalizer
from mimic4models import metrics
from mimic4models import keras_utils
from mimic4models import common_utils

from keras.callbacks import ModelCheckpoint, CSVLogger

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

def expected_calibration_error(y_true, y_prob, n_bins=10):

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1

    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        mask = binids == i

        if np.sum(mask) > 0:
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])

            gap = abs(acc - conf)

            ece += (np.sum(mask) / len(y_true)) * gap
            mce = max(mce, gap)

    return ece, mce



parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/in-hospital-mortality/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
args = parser.parse_args()
print(args)

if args.small_part:
    args.save_every = 2**30

#target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')
target_repl = (args.target_repl_coef > 0.0)
# Build readers, discretizers, normalizers
train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                         listfile=os.path.join(args.data, 'train_listfile.csv'),
                                         period_length=48.0)

val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                       listfile=os.path.join(args.data, 'val_listfile.csv'),
                                       period_length=48.0)

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
feature_names = np.array(discretizer_header)

train_raw = utils.load_data(train_reader, discretizer, None, args.small_part)

X_train = train_raw[0]
print("Raw HR:", X_train[12242, 28, 52])
print("Raw RR:", X_train[2401, 4, 62])

ret = utils.load_data(
    train_reader,
    discretizer,
    None,
    args.small_part,
    return_names=True
)

print(ret["names"][12242])   # Heart rate outlier
print(ret["names"][2401])    # Respiratory rate outlier

print("Fitting normalizer...")
cont_channels = [
    i for i, x in enumerate(discretizer_header)
    if '->' not in x and not x.startswith('mask->')
]
normalizer = Normalizer(fields=cont_channels)
normalizer.fit(X_train)

normalizer_path = os.path.join(args.output_dir, "normalizer.pkl")
print("CHECKPOINT A PASSED")
normalizer.save_params(normalizer_path)
print("CHECKPOINT B PASSED")
print("Saved new normalizer:", normalizer_path)

normalizer.load_params(normalizer_path)
print("CHECKPOINT C PASSED")
assert np.all(np.isfinite(normalizer._means))
assert np.all(np.isfinite(normalizer._stds))
assert np.all(normalizer._stds > 0)
print("CHECKPOINT D PASSED")
print("Normalizer loaded.")
print("CHECKPOINT E PASSED")
print("Header length:", len(discretizer_header))
print("Continuous channels:", len(cont_channels))
print("Normalizer means:", len(normalizer._means))
print("Normalizer stds:", len(normalizer._stds))
print("CHECKPOINT F PASSED")

print("\nCONTINUOUS FEATURES:")
for idx in cont_channels:
    print(idx, discretizer_header[idx])

print("Total features:", X_train.shape[2])
print("Continuous features:", len(cont_channels))

print("CHECKPOINT G PASSED")
train_raw = utils.load_data(train_reader, discretizer, normalizer, args.small_part)

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['task'] = 'ihm'
args_dict['target_repl'] = target_repl

preprocessing_pipeline = {
    "discretizer": discretizer,
    "normalizer": normalizer,
    "discretizer_header": discretizer_header,
    "cont_channels": cont_channels,
    "feature_names": feature_names,
    "task": "ihm",
    "timestep": args.timestep,
    "imputation": args.imputation,
}

pipeline_path = os.path.join(
    args.output_dir,
    "preprocessing_pipeline.pkl"
)

joblib.dump(preprocessing_pipeline, pipeline_path)

print("==> saved preprocessing pipeline to:", pipeline_path)
# Build the model
print("==> using model {}".format(args.network))
model_module = imp.load_source(os.path.basename(args.network), args.network)
model = model_module.Network(**args_dict)
suffix = ".bs{}{}{}.ts{}{}".format(args.batch_size,
                                   ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                   ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                   args.timestep,
                                   ".trc{}".format(args.target_repl_coef) if args.target_repl_coef > 0 else "")
model.final_name = args.prefix + model.say_name() + suffix
print("==> model.final_name:", model.final_name)


# Compile the model
print("==> compiling the model")
optimizer_config = {'class_name': args.optimizer,
                    'config': {'learning_rate': args.lr,
                               'beta_1': args.beta_1}}

# NOTE: one can use binary_crossentropy even for (B, T, C) shape.
#       It will calculate binary_crossentropies for each class
#       and then take the mean over axis=-1. Tre results is (B, T).
if target_repl:
    loss = ['binary_crossentropy'] * 2
    loss_weights = [1 - args.target_repl_coef, args.target_repl_coef]
else:
    loss = 'binary_crossentropy'
    loss_weights = None

model.compile(optimizer=optimizer_config,
              loss=loss,
              loss_weights=loss_weights)
model.summary()

# Load model weights
n_trained_chunks = 0
if args.load_state != "":
    model.load_weights(args.load_state)
    n_trained_chunks = int(re.match(".*epoch([0-9]+).*", args.load_state).group(1))


# Read data
train_raw = utils.load_data(train_reader, discretizer, normalizer, args.small_part)
X = train_raw[0]
for idx in cont_channels:
    col = X[:, :, idx]
    print(
        discretizer_header[idx],
        np.nanmean(col),
        np.nanstd(col)
    )
val_raw = utils.load_data(val_reader, discretizer, normalizer, args.small_part)

print("=== NORMALIZATION CONSISTENCY CHECK ===")
print("X shape:", X.shape)
print("means:", len(normalizer._means))
print("stds:", len(normalizer._stds))
print("cont_channels:", len(cont_channels))
print("feature_names:", len(feature_names))
print("========================================")

for j in range(X.shape[2]):
    col = X[:, :, j]

    max_val = np.nanmax(col)
    min_val = np.nanmin(col)

    if max_val > 1000 or min_val < -1000:
        mask = np.abs(col) > 100
        indices = np.argwhere(mask)

        print(
            "BAD FEATURE:",
            j,
            feature_names[j],
            "min =", min_val,
            "max =", max_val,
            "NaNs =", np.isnan(col).sum(),
            "Values with |x| > 100 =", np.sum(mask),
            "Extreme indices:", indices[:10]  # show first 10
        )

print("FINAL CHECK:",
      X.shape,
      len(normalizer._means),
      len(cont_channels))

if target_repl:
    T = train_raw[0][0].shape[0]

    def extend_labels(data):
        data = list(data)
        labels = np.array(data[1])  # (B,)
        data[1] = [labels, None]
        data[1][1] = np.expand_dims(labels, axis=-1).repeat(T, axis=1)  # (B, T)
        data[1][1] = np.expand_dims(data[1][1], axis=-1)  # (B, T, 1)
        return data

    train_raw = extend_labels(train_raw)
    val_raw = extend_labels(val_raw)

if args.mode == 'train':

    # Prepare training
    path = os.path.join(args.output_dir, 'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.keras')

    metrics_callback = keras_utils.InHospitalMortalityMetrics(train_data=train_raw,
                                                              val_data=val_raw,
                                                              target_repl=(args.target_repl_coef > 0),
                                                              batch_size=args.batch_size,
                                                              verbose=args.verbose)
    # make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    saver = ModelCheckpoint(path, monitor='val_loss',save_best_only=True, verbose=1)

    keras_logs = os.path.join(args.output_dir, 'keras_logs')
    if not os.path.exists(keras_logs):
        os.makedirs(keras_logs)
    csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                           append=True, separator=';')

    print("==> training")
    train_x = tf.convert_to_tensor(train_raw[0], dtype=tf.float32)
    val_x = tf.convert_to_tensor(val_raw[0], dtype=tf.float32)

    if target_repl:
        train_y = [
            tf.convert_to_tensor(train_raw[1][0], dtype=tf.float32),
            tf.convert_to_tensor(train_raw[1][1], dtype=tf.float32)
        ]

        val_y = [
            tf.convert_to_tensor(val_raw[1][0], dtype=tf.float32),
            tf.convert_to_tensor(val_raw[1][1], dtype=tf.float32)
        ]
    else:
        train_y = tf.convert_to_tensor(train_raw[1], dtype=tf.float32)
        val_y = tf.convert_to_tensor(val_raw[1], dtype=tf.float32)

    model.fit(x=train_x,
        y=train_y,
        validation_data=(val_x, val_y),
        epochs=n_trained_chunks + args.epochs,
        initial_epoch=n_trained_chunks,
        callbacks=[metrics_callback, saver, csv_logger],
        shuffle=True,
        verbose=args.verbose,
        batch_size=args.batch_size)

elif args.mode == 'test':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    #del train_raw
    #del val_raw

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=48.0)
    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                          return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]

    test_snapshot = {
    "X": data,          # already discretized + normalized
    "y": labels,
    "names": names
    }

    snapshot_path = os.path.join(args.output_dir, "test_snapshot.pkl")
    joblib.dump(test_snapshot, snapshot_path)

    print("Saved test snapshot to:", snapshot_path)
    test_icu_map = build_icu_map(os.path.join(args.data, "../root"))
    test_icu = [test_icu_map.get(n, "UNKNOWN") for n in names]
    test_icu = np.array(test_icu)

    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)

    if target_repl:
        predictions = predictions[0]

    predictions = np.array(predictions).ravel()

    # -----------------------------
    # Test metrics
    # -----------------------------

    #test_metrics = metrics.print_metrics_binary(labels, predictions, "Test Confusion Matrix", result_dir=os.path.join(args.output_dir, "results"))

    y_true = np.array(labels)
    y_prob = predictions


    # -----------------------------
    # ICU Calibration curve
    # -----------------------------

    unique_icus = np.unique(test_icu)
    result_dir = os.path.join(args.output_dir, "results")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    plt.figure(figsize=(8,6))

    for icu in unique_icus:
        idx = (test_icu == icu)

        if np.sum(idx) < 100:
            continue

        prob_true, prob_pred = calibration_curve(
            y_true[idx],
            y_prob[idx],
           n_bins=10,
            strategy="quantile"
        )

        if len(prob_true) > 1:
            plt.plot(prob_pred, prob_true, marker="o", label=icu)

    plt.plot([0,1],[0,1],"--",color="black")
    plt.title("Channel-wise LSTM Calibration by ICU (Test)")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(alpha=0.3)

    plt.savefig(os.path.join(result_dir, "channelwise_lstm_calibration_by_icu.png"), bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Validation predictions
    # -----------------------------

    val_data = val_raw[0]
    if target_repl:
        val_labels = np.array(val_raw[1][0]).astype(np.float32)
    else:
        val_labels = np.asarray(val_raw[1]).ravel().astype(np.float32)
    val_predictions = model.predict(
        val_data,
        batch_size=args.batch_size,
        verbose=1
    )

    if target_repl:
        val_predictions = val_predictions[0]

    val_predictions = np.array(val_predictions).ravel()

    # -----------------------------
    # Threshold optimization
    # -----------------------------

    precision, recall, thresholds = precision_recall_curve(
        val_labels,
        val_predictions
    )

    f1_scores = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-8
    )

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print("\nBest threshold:", best_threshold)
    print("Best validation F1:", f1_scores[best_idx])

    threshold_info = {
        "best_threshold": float(best_threshold),
        "best_validation_f1": float(f1_scores[best_idx])
    }

    with open(os.path.join(result_dir, "threshold_info.json"), "w") as f:
        json.dump(threshold_info, f, indent=4)

    # -----------------------------
    # Train predictions
    # -----------------------------

    train_predictions = model.predict(
        train_raw[0],
        batch_size=args.batch_size,
        verbose=1
    )

    if target_repl:
        train_predictions = train_predictions[0]

    train_predictions = np.array(train_predictions).ravel()

    if target_repl:
        train_labels = np.asarray(train_raw[1][0]).ravel()
    else:
        train_labels = np.asarray(train_raw[1]).ravel()
    train_metrics = metrics.print_metrics_binary(
        train_labels,
        train_predictions,
        "Train Confusion Matrix",
        result_dir,
        threshold=best_threshold
    )

    # -----------------------------
    # Validation metrics
    # -----------------------------

    val_metrics = metrics.print_metrics_binary(
        val_labels,
        val_predictions,
        "Validation Confusion Matrix",
        result_dir,
        threshold=best_threshold
    )

    # -----------------------------
    # Test metrics
    # -----------------------------

    test_metrics = metrics.print_metrics_binary(
        y_true,
        y_prob,
        "Test Confusion Matrix",
        result_dir,
        threshold=best_threshold
    )

    # -----------------------------
    # Calibration metrics
    # -----------------------------

    brier = brier_score_loss(y_true, y_prob)

    ece, mce = expected_calibration_error(
        y_true,
        y_prob,
        n_bins=10
    )

    # -----------------------------
    # Train calibration
    # -----------------------------

    train_brier = brier_score_loss(
        train_labels,
        train_predictions
    )

    train_ece, train_mce = expected_calibration_error(
        train_labels,
        train_predictions
    )

    # -----------------------------
    # Validation calibration
    # -----------------------------

    val_brier = brier_score_loss(
        val_labels,
        val_predictions
    )

    val_ece, val_mce = expected_calibration_error(
        val_labels,
        val_predictions
    )

    # -----------------------------
    # Calibration curve
    # -----------------------------

    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=10,
        strategy="quantile"
    )

    plt.figure()

    plt.plot(
        prob_pred,
        prob_true,
        marker="o",
        label="Channel-wise LSTM"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfectly calibrated"
    )

    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Curve - Test Set")
    plt.legend()

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    plt.savefig(
        os.path.join(result_dir, "channelwise_lstm_calibration_curve_test.png"),
        bbox_inches="tight"
    )

    plt.close()

    # -----------------------------
    # ROC + PR curves
    # -----------------------------
    metrics.plot_roc_curve(
        y_true,
        y_prob,
        "Channel-wise LSTM AUROC - Test Set",
        result_dir
    )

    metrics.plot_pr_curve(
        y_true,
        y_prob,
        "Channel-wise LSTM AUPRC - Test Set",
        result_dir
    )

    # -----------------------------
    # SHAP
    # -----------------------------

    # small background sample
    background = train_raw[0]
    background = background[
        np.random.choice(background.shape[0], 50, replace=False)
    ]

    # test sample
    X_test_sample = data[
        np.random.choice(data.shape[0], 100, replace=False)
    ]

    # shapes
    n_timesteps = X_test_sample.shape[1]
    n_features = X_test_sample.shape[2]

    # flatten for SHAP
    # aggregate over time first
    background_agg = np.mean(background, axis=1)   # (N, F)
    X_test_agg = np.mean(X_test_sample, axis=1)    # (N, F)

    def predict_fn(x):
        # expand aggregated features back to pseudo-sequence
        x_seq = np.repeat(
            x[:, np.newaxis, :],
            n_timesteps,
            axis=1
        )

        pred = model.predict(x_seq, verbose=0)

        if target_repl:
            pred = pred[0]

        return np.asarray(pred).ravel()

    explainer = shap.KernelExplainer(
        predict_fn,
        background_agg
    )

    shap_values = explainer.shap_values(
       X_test_agg,
        nsamples=100
    )

    shap_values = np.array(shap_values)

    plt.figure(figsize=(12,8))

    shap.summary_plot(
        shap_values,
        X_test_agg,
        feature_names=feature_names,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(result_dir, "channelwise_lstm_shap_summary.png"),
        dpi=300
    )

    plt.close()
    # -----------------------------
    # Add calibration metrics
    # -----------------------------

    test_metrics = {k: float(v) for k, v in test_metrics.items()}
    train_metrics = {k: float(v) for k, v in train_metrics.items()}
    val_metrics = {k: float(v) for k, v in val_metrics.items()}

    test_metrics["brier_score"] = float(brier)
    test_metrics["ece"] = float(ece)
    test_metrics["mce"] = float(mce)

    train_metrics["brier_score"] = float(train_brier)
    train_metrics["ece"] = float(train_ece)
    train_metrics["mce"] = float(train_mce)

    val_metrics["brier_score"] = float(val_brier)
    val_metrics["ece"] = float(val_ece)
    val_metrics["mce"] = float(val_mce)

    # -----------------------------
    # Summary table
    # -----------------------------

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
        "minpse": "min(P,Se)",
        "brier_score": "Brier Score",
        "ece": "Expected Calibration Error",
        "mce": "Maximum Calibration Error"
    })

    results_df = results_df.round(4)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)

    print("\nModel Performance Summary:\n")
    print(results_df)

    # -----------------------------
    # Save metrics
    # -----------------------------
    json.dump(
        train_metrics,
        open(os.path.join(result_dir, "train_channelwise_lstm.json"), "w")
    )

    json.dump(
        val_metrics,
        open(os.path.join(result_dir, "val_channelwise_lstm.json"), "w")
    )

    json.dump(
        test_metrics,
        open(os.path.join(result_dir, "test_channelwise_lstm.json"), "w")
    )

    # -----------------------------
    # Save predictions
    # -----------------------------

    path = os.path.join(
        args.output_dir,
        "test_predictions",
        os.path.basename(args.load_state)
    ) + ".csv"

    utils.save_results(names, predictions, labels, path)
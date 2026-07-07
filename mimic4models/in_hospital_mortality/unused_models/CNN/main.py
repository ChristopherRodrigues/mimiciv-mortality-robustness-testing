from __future__ import absolute_import
from __future__ import print_function

from mimic4benchmark.readers import InHospitalMortalityReader
from mimic4models import common_utils
from mimic4models.metrics import print_metrics_binary
from mimic4models.in_hospital_mortality.utils import save_results
from sklearn.preprocessing import StandardScaler
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
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scikeras.wrappers import KerasClassifier

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

def read_and_extract_features(reader, period, features, icu_map):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    y = ret['y']
    names = ret['name']
    icu = [icu_map.get(n, "UNKNOWN") for n in names]
    return (X, y, names, ret['header'], icu)

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

def create_cnn_model(input_dim, filters=64, kernel_size=3, num_conv_blocks=2, 
                     dense_units=128, dropout_rate=0.3, l2_reg=0.001, learning_rate=0.001):
    """
    CNN for 1D tabular data.
    Treats features as a sequence where local patterns can be meaningful.
    """
    model = keras.Sequential()
    
    # Reshape input: (batch, features) -> (batch, features, 1)
    model.add(layers.Reshape((input_dim, 1), input_shape=(input_dim,)))
    
    # Convolutional blocks
    for i in range(num_conv_blocks):
        model.add(layers.Conv1D(
            filters=filters * (2**i),  # increase filters in deeper layers
            kernel_size=kernel_size,
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(l2_reg)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(dropout_rate))
    
    # Global pooling to aggregate spatial information
    model.add(layers.GlobalAveragePooling1D())
    
    # Dense layers
    model.add(layers.Dense(dense_units, activation='relu', 
                          kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(dense_units // 2, activation='relu',
                          kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.Dropout(dropout_rate / 2))
    
    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['AUC', 'Precision', 'Recall']
    )
    
    return model

class CNNClassifierWrapper(ClassifierMixin, BaseEstimator):
    """
    Wrapper to make Keras CNN compatible with sklearn's CalibratedClassifierCV
    """
    def __init__(self, model=None, epochs=100, batch_size=64, verbose=0):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.history_ = None
        
    def fit(self, X, y):
        if self.model is None:
            raise ValueError("Model not initialized")
            
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=self.verbose
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=self.verbose
        )
        
        # Compute class weights for imbalanced data
        class_weights = {
            0: len(y) / (2 * np.sum(y == 0)),
            1: len(y) / (2 * np.sum(y == 1))
        }
        
        self.history_ = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            class_weight=class_weights,
            callbacks=[early_stop, reduce_lr],
            verbose=self.verbose
        )
        
        return self
    
    def predict_proba(self, X):
        pred = self.model.predict(X, verbose=0)
        # Return shape (n_samples, 2) for binary classification
        return np.hstack([1 - pred, pred])
    
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', type=str, default='all', 
                        choices=['first4days', 'first8days', 'last12hours', 'first25percent', 'first50percent', 'all'])
    parser.add_argument('--features', type=str, default='all', 
                        choices=['all', 'len', 'all_but_len'])
    parser.add_argument('--data', type=str, 
                        default=os.path.join(os.path.dirname(__file__), '../../../data/in-hospital-mortality/'))
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    print(args)

    # Read data
    train_reader = InHospitalMortalityReader(
        dataset_dir=os.path.join(args.data, 'train'),
        listfile=os.path.join(args.data, 'train_listfile.csv'),
        period_length=48.0
    )
    val_reader = InHospitalMortalityReader(
        dataset_dir=os.path.join(args.data, 'train'),
        listfile=os.path.join(args.data, 'val_listfile.csv'),
        period_length=48.0
    )
    test_reader = InHospitalMortalityReader(
        dataset_dir=os.path.join(args.data, 'test'),
        listfile=os.path.join(args.data, 'test_listfile.csv'),
        period_length=48.0
    )
    
    icu_map = build_icu_map(os.path.join(args.data, "../root"))
    
    print('Reading data and extracting features ...')
    (train_X, train_y, train_names, train_feature_header, train_icu) = read_and_extract_features(
        train_reader, args.period, args.features, icu_map
    )
    (val_X, val_y, val_names, val_feature_header, val_icu) = read_and_extract_features(
        val_reader, args.period, args.features, icu_map
    )
    (test_X, test_y, test_names, test_feature_header, test_icu) = read_and_extract_features(
        test_reader, args.period, args.features, icu_map
    )
    
    print('  train data shape = {}'.format(train_X.shape))
    print('  validation data shape = {}'.format(val_X.shape))
    print('  test data shape = {}'.format(test_X.shape))
    
    print("\nICU distribution (train):", Counter(train_icu))
    print("ICU distribution (val):", Counter(val_icu))
    print("ICU distribution (test):", Counter(test_icu))

    # One-hot encode ICU units
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(np.array(train_icu).reshape(-1, 1))
    train_icu_enc = encoder.transform(np.array(train_icu).reshape(-1, 1))
    val_icu_enc = encoder.transform(np.array(val_icu).reshape(-1, 1))
    test_icu_enc = encoder.transform(np.array(test_icu).reshape(-1, 1))

    # Remove empty columns
    non_empty_cols = ~np.all(np.isnan(train_X), axis=0)
    train_X = train_X[:, non_empty_cols]
    val_X = val_X[:, non_empty_cols]
    test_X = test_X[:, non_empty_cols]
    
    # Impute missing values
    print('Imputing missing values ...')
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(train_X)
    train_X = np.array(imputer.transform(train_X), dtype=np.float32)
    val_X = np.array(imputer.transform(val_X), dtype=np.float32)
    test_X = np.array(imputer.transform(test_X), dtype=np.float32)
    
    # Build feature names
    feature_names = np.array(build_feature_names(train_feature_header))
    feature_names = feature_names[non_empty_cols]

    # Variance threshold
    print("\nRemoving low-variance features...")
    print("Features before VarianceThreshold:", train_X.shape[1])
    selector = VarianceThreshold(threshold=0.01)
    train_X = selector.fit_transform(train_X)
    val_X = selector.transform(val_X)
    test_X = selector.transform(test_X)
    feature_names = feature_names[selector.get_support()]
    print("Features after VarianceThreshold:", train_X.shape[1])

    # Correlation filtering
    train_X_df = pd.DataFrame(train_X, columns=feature_names)
    val_X_df = pd.DataFrame(val_X, columns=feature_names)
    test_X_df = pd.DataFrame(test_X, columns=feature_names)
    
    print("\nRemoving highly correlated features...")
    corr_matrix = train_X_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    print(f"Removing {len(to_drop)} correlated features")
    
    train_X_df = train_X_df.drop(columns=to_drop)
    val_X_df = val_X_df.drop(columns=to_drop)
    test_X_df = test_X_df.drop(columns=to_drop)
    print("Remaining features:", train_X_df.shape[1])
    
    feature_names = train_X_df.columns.to_numpy()
    train_X_reduced = train_X_df.values
    val_X_reduced = val_X_df.values
    test_X_reduced = test_X_df.values

    # Normalization
    print('Normalizing data for CNN ...')
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X_reduced)
    val_X_scaled = scaler.transform(val_X_reduced)
    test_X_scaled = scaler.transform(test_X_reduced)

    # Append ICU features
    train_X_scaled = np.hstack([train_X_scaled, train_icu_enc])
    val_X_scaled = np.hstack([val_X_scaled, val_icu_enc])
    test_X_scaled = np.hstack([test_X_scaled, test_icu_enc])
    
    # Update feature names
    icu_names = encoder.get_feature_names_out(["ICU"])
    feature_names = np.concatenate([feature_names, icu_names])

    # Convert y to numpy arrays
    train_y = np.array(train_y)
    val_y = np.array(val_y)
    test_y = np.array(test_y)

    file_name = '{}.{}.cnn'.format(args.period, args.features)
    result_dir = os.path.join(args.output_dir, 'results')
    common_utils.create_directory(result_dir)
    common_utils.create_directory(os.path.join(args.output_dir, 'predictions'))

    input_dim = train_X_scaled.shape[1]
    print(f"\nInput dimension: {input_dim}")

    # Hyperparameter search space for CNN
    param_dist = {
        'model__filters': [32, 64, 128],
        'model__kernel_size': [3, 5, 7],
        'model__num_conv_blocks': [2, 3],
        'model__dense_units': [64, 128, 256],
        'model__dropout_rate': uniform(0.2, 0.4),
        'model__l2_reg': uniform(1e-5, 1e-2),
        'model__learning_rate': uniform(1e-4, 1e-2),
        'batch_size': [32, 64, 128],
        'epochs': [100]  # Fixed, early stopping will control actual training
    }

    # Create KerasClassifier wrapper for sklearn compatibility
    def create_model_wrapper(filters=64, kernel_size=3, num_conv_blocks=2,
                            dense_units=128, dropout_rate=0.3, l2_reg=0.001, 
                            learning_rate=0.001):
        return create_cnn_model(
            input_dim=input_dim,
            filters=filters,
            kernel_size=kernel_size,
            num_conv_blocks=num_conv_blocks,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            learning_rate=learning_rate
        )

    keras_clf = KerasClassifier(
        model=create_model_wrapper,
        verbose=0
    )

    print("\nStarting hyperparameter search...")
    random_search = RandomizedSearchCV(
        estimator=keras_clf,
        param_distributions=param_dist,
        n_iter=15,  # Reduced because CNN training is slower
        cv=3,
        scoring='average_precision',
        verbose=2,
        random_state=42,
        n_jobs=1  # Keras models can't be parallelized this way
    )

    random_search.fit(train_X_scaled, train_y)
    
    print("\nBest parameters:", random_search.best_params_)
    print("Best score:", random_search.best_score_)

    # Get best model
    best_cnn = random_search.best_estimator_

    # Split validation set for calibration
    val_X_calib, val_X_thresh, val_y_calib, val_y_thresh = train_test_split(
        val_X_scaled, val_y, test_size=0.5, random_state=42, stratify=val_y
    )

    # Wrap best model for calibration
    best_params = random_search.best_params_
    
    # Extract model parameters
    model_params = {k.replace('model__', ''): v for k, v in best_params.items() if k.startswith('model__')}
    
    final_model = create_cnn_model(input_dim=input_dim, **model_params)
    cnn_wrapper = CNNClassifierWrapper(
        model=final_model,
        epochs=best_params.get('epochs', 100),
        batch_size=best_params.get('batch_size', 64),
        verbose=1
    )

    # Fit on full training set
    print("\nTraining final model on full training set...")
    cnn_wrapper.fit(train_X_scaled, train_y)

    # Plot training history
    if cnn_wrapper.history_ is not None:
        history = cnn_wrapper.history_.history
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        axes[0, 0].plot(history['loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # AUC
        axes[0, 1].plot(history['auc'], label='Train AUC')
        axes[0, 1].plot(history['val_auc'], label='Val AUC')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].set_title('Training and Validation AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(history['precision'], label='Train Precision')
        axes[1, 0].plot(history['val_precision'], label='Val Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Training and Validation Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(history['recall'], label='Train Recall')
        axes[1, 1].plot(history['val_recall'], label='Val Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_title('Training and Validation Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "cnn_training_history.png"), dpi=150)
        plt.close()

    # Calibration (manual approach to avoid sklearn deprecation)
    print("\nCalibrating model...")
    from sklearn.isotonic import IsotonicRegression
    
    val_probs_uncalib = cnn_wrapper.predict_proba(val_X_calib)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(val_probs_uncalib, val_y_calib)
    
    class CalibratedCNN:
        def __init__(self, cnn, calibrator):
            self.cnn = cnn
            self.calibrator = calibrator
        
        def predict_proba(self, X):
            probs = self.cnn.predict_proba(X)[:, 1]
            calibrated = self.calibrator.predict(probs)
            return np.column_stack([1 - calibrated, calibrated])
        
        def predict(self, X):
            return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
    
    calibrated_cnn = CalibratedCNN(cnn_wrapper, calibrator)

    # Save models
    cnn_wrapper.model.save(os.path.join(args.output_dir, "cnn_model.keras"))
    joblib.dump(calibrator, os.path.join(args.output_dir, "cnn_calibrator.pkl"))
    joblib.dump(scaler, os.path.join(args.output_dir, "cnn_scaler.pkl"))

    # Predictions
    prediction = calibrated_cnn.predict_proba(test_X_scaled)[:, 1]
    y_prob = prediction
    y_true = test_y

    # Threshold optimization
    print("\nOptimizing classification threshold...")
    val_probs = calibrated_cnn.predict_proba(val_X_thresh)[:, 1]
    precision, recall, thresholds = precision_recall_curve(val_y_thresh, val_probs)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    print("Best threshold from validation set:", best_threshold)
    print("Best validation F1:", f1_scores[best_idx])

    # Calibration curve (overall)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label="CNN")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Curve - Test Set")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(result_dir, "cnn_calibration_curve_test.png"), bbox_inches="tight")
    plt.close()

    # Probability distribution
    plt.figure()
    plt.hist(y_prob, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Distribution of Predicted Probabilities")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(result_dir, "cnn_prob_distribution.png"), bbox_inches="tight")
    plt.close()

    # Calibration by ICU
    test_icu = np.array(test_icu)
    unique_icus = np.unique(test_icu)
    plt.figure(figsize=(8, 6))
    
    for icu in unique_icus:
        idx = (test_icu == icu)
        if np.sum(idx) < 100:
            continue
        prob_true, prob_pred = calibration_curve(
            test_y[idx], prediction[idx], n_bins=10, strategy="quantile"
        )
        if len(prob_true) > 1:
            plt.plot(prob_pred, prob_true, marker="o", linewidth=1.5, label=icu)
    
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Curves by ICU (Test Set)")
    plt.legend(fontsize=8, loc="best", ncol=2)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(result_dir, "cnn_calibration_all_ICUs.png"), bbox_inches="tight")
    plt.close()

    # ROC and PR curves
    plot_roc_curve(y_true, y_prob, "CNN AUROC - Test Set", result_dir)
    plot_pr_curve(y_true, y_prob, "CNN AUPRC - Test Set", result_dir)

    # SHAP analysis
    print("\nComputing SHAP values...")
    background = shap.sample(train_X_scaled, 100)
    explainer = shap.KernelExplainer(cnn_wrapper.predict_proba, background)
    
    sample_idx = np.random.choice(len(test_X_scaled), min(200, len(test_X_scaled)), replace=False)
    X_sample = test_X_scaled[sample_idx]
    
    shap_values = explainer.shap_values(X_sample)
    
    # Handle binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    
    # SHAP plots
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(result_dir, "cnn_test_shap_summary.png"), bbox_inches="tight")
    plt.close()
    
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.savefig(os.path.join(result_dir, "cnn_test_shap_bar.png"), bbox_inches="tight")
    plt.close()

    # Metrics
    train_metrics = print_metrics_binary(
        train_y, calibrated_cnn.predict_proba(train_X_scaled), 
        "Train Confusion Matrix", result_dir, threshold=best_threshold
    )
    val_metrics = print_metrics_binary(
        val_y_thresh, calibrated_cnn.predict_proba(val_X_thresh), 
        "Validation Confusion Matrix", result_dir, threshold=best_threshold
    )
    test_metrics = print_metrics_binary(
        test_y, calibrated_cnn.predict_proba(test_X_scaled), 
        "Test Confusion Matrix", result_dir, threshold=best_threshold
    )

    train_metrics = {k: float(v) for k, v in train_metrics.items()}
    val_metrics = {k: float(v) for k, v in val_metrics.items()}
    test_metrics = {k: float(v) for k, v in test_metrics.items()}

    json.dump(train_metrics, open(os.path.join(result_dir, f"train_{file_name}.json"), "w"))
    json.dump(val_metrics, open(os.path.join(result_dir, f"val_{file_name}.json"), "w"))
    json.dump(test_metrics, open(os.path.join(result_dir, f"test_{file_name}.json"), "w"))

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
    main()
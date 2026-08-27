"""MSI/MSS classifier comparison for patient-level genomic features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler


COHORTS = {
    "COAD_READ": ["COAD", "READ"],
    "UCEC": ["UCEC"],
    "STAD": ["STAD"],
}
MODEL_FEATURES = {
    "TMB only": ["TMB_mutations_per_mb"],
    "Model features": None,
    "TMB + model features": None,
}
GENOME_LENGTH_MB = 3088.27


def load_msi_labels(
    coad_path: str | Path,
    ucec_path: str | Path,
    stad_path: str | Path,
) -> pd.DataFrame:
    coad = pd.read_csv(coad_path)
    coad = coad.dropna(subset=["MSI_SCORE_MANTIS", "MSI_SENSOR_SCORE"]).copy()
    coad["mantis_status"] = np.where(coad["MSI_SCORE_MANTIS"] > 0.4, "MSI-H", "MSS")
    coad["sensor_status"] = np.where(coad["MSI_SENSOR_SCORE"] > 3.5, "MSI-H", "MSS")
    coad = coad[coad["mantis_status"] == coad["sensor_status"]].copy()
    coad["msi_status"] = coad["mantis_status"]

    ucec = pd.read_csv(ucec_path)
    ucec_cols = ["MSI_STATUS_5_MARKER_CALL", "MSI_STATUS_7_MARKER_CALL"]
    ucec[ucec_cols] = ucec[ucec_cols].apply(
        lambda column: column.astype("string").str.strip().str.upper()
    )
    valid_calls = {"MSS", "MSI-L", "MSI-H"}
    ucec = ucec[ucec[ucec_cols].isin(valid_calls).all(axis=1)].copy()
    for column in ucec_cols:
        ucec[column] = ucec[column].replace("MSI-L", "MSS")
    ucec = ucec[ucec[ucec_cols[0]] == ucec[ucec_cols[1]]].copy()
    ucec["msi_status"] = ucec[ucec_cols[0]]

    stad = pd.read_csv(stad_path)
    stad["msi_status"] = (
        stad["MSI_STATUS"].astype("string").str.strip().str.upper().replace("MSI-L", "MSS")
    )
    stad = stad[stad["msi_status"].isin(["MSS", "MSI-H"])].copy()

    labels = pd.concat(
        [
            coad[["patient_barcode", "msi_status"]],
            ucec[["patient_barcode", "msi_status"]],
            stad[["patient_barcode", "msi_status"]],
        ],
        ignore_index=True,
    ).rename(columns={"patient_barcode": "patient_id"})
    return labels.drop_duplicates("patient_id")


def load_tmb(tmb_dir: str | Path) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(tmb_dir).glob("*_tmb.tsv")):
        try:
            frame = pd.read_csv(
                path,
                sep="\t",
                usecols=["bcr_patient_barcode", "total_mutations", "cancer"],
            )
        except (pd.errors.EmptyDataError, ValueError):
            continue
        frames.append(frame)
    tmb = pd.concat(frames, ignore_index=True).rename(
        columns={"bcr_patient_barcode": "patient_id", "cancer": "cancer_type"}
    )
    tmb["TMB_mutations_per_mb"] = (
        pd.to_numeric(tmb["total_mutations"], errors="coerce") / GENOME_LENGTH_MB
    )
    return tmb[["patient_id", "cancer_type", "TMB_mutations_per_mb"]].drop_duplicates(
        ["patient_id", "cancer_type"]
    )


def classifier_pipeline() -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        MaxAbsScaler(),
        LogisticRegression(
            penalty="l2",
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=42,
        ),
    )


def evaluate_model(
    frame: pd.DataFrame,
    feature_cols: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, float], np.ndarray]:
    x = frame[feature_cols]
    y = frame["msi_label"].to_numpy()
    probabilities = np.zeros(len(frame), dtype=float)

    for train_index, test_index in splits:
        model = classifier_pipeline()
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            model.fit(x.iloc[train_index], y[train_index])
            probabilities[test_index] = model.predict_proba(x.iloc[test_index])[:, 1]

    predictions = (probabilities >= 0.5).astype(int)
    negative = y == 0
    positive = y == 1
    metrics = {
        "n": len(y),
        "n_mss": int(negative.sum()),
        "n_msi_h": int(positive.sum()),
        "roc_auc": roc_auc_score(y, probabilities),
        "average_precision": average_precision_score(y, probabilities),
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "f1": f1_score(y, predictions),
        "sensitivity": float(predictions[positive].mean()),
        "specificity": float((predictions[negative] == 0).mean()),
    }
    return metrics, probabilities


def residualize_features(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_tmb: pd.Series,
    test_tmb: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove linear log-TMB effects using parameters learned on training data."""
    train_values = train_features.to_numpy(dtype=float)
    test_values = test_features.to_numpy(dtype=float)
    train_values[~np.isfinite(train_values)] = np.nan
    test_values[~np.isfinite(test_values)] = np.nan
    usable_columns = ~np.isnan(train_values).all(axis=0)
    train_values = train_values[:, usable_columns]
    test_values = test_values[:, usable_columns]
    train_medians = np.nanmedian(train_values, axis=0)
    train_values = np.where(np.isnan(train_values), train_medians, train_values)
    test_values = np.where(np.isnan(test_values), train_medians, test_values)
    feature_scales = np.max(np.abs(train_values), axis=0)
    feature_scales[feature_scales == 0] = 1
    train_values = train_values / feature_scales
    test_values = test_values / feature_scales

    train_log_tmb = np.log1p(train_tmb.to_numpy(dtype=float))
    test_log_tmb = np.log1p(test_tmb.to_numpy(dtype=float))
    tmb_mean = train_log_tmb.mean()
    centered_tmb = train_log_tmb - tmb_mean
    denominator = np.sum(centered_tmb ** 2)

    feature_means = train_values.mean(axis=0)
    slopes = (
        np.sum(
            centered_tmb[:, np.newaxis] * (train_values - feature_means),
            axis=0,
        )
        / denominator
        if denominator > 0
        else np.zeros(train_values.shape[1])
    )
    train_residuals = train_values - (
        feature_means + np.outer(train_log_tmb - tmb_mean, slopes)
    )
    test_residuals = test_values - (
        feature_means + np.outer(test_log_tmb - tmb_mean, slopes)
    )
    return train_residuals, test_residuals


def evaluate_residualized_model(
    frame: pd.DataFrame,
    feature_cols: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, float], np.ndarray]:
    y = frame["msi_label"].to_numpy()
    probabilities = np.zeros(len(frame), dtype=float)

    for train_index, test_index in splits:
        train_residuals, test_residuals = residualize_features(
            frame.iloc[train_index][feature_cols],
            frame.iloc[test_index][feature_cols],
            frame.iloc[train_index]["TMB_mutations_per_mb"],
            frame.iloc[test_index]["TMB_mutations_per_mb"],
        )
        model = classifier_pipeline()
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            model.fit(train_residuals, y[train_index])
            probabilities[test_index] = model.predict_proba(test_residuals)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)
    negative = y == 0
    positive = y == 1
    metrics = {
        "n": len(y),
        "n_mss": int(negative.sum()),
        "n_msi_h": int(positive.sum()),
        "roc_auc": roc_auc_score(y, probabilities),
        "average_precision": average_precision_score(y, probabilities),
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "f1": f1_score(y, predictions),
        "sensitivity": float(predictions[positive].mean()),
        "specificity": float((predictions[negative] == 0).mean()),
    }
    return metrics, probabilities


def run_msi_benchmarks(
    feature_df: pd.DataFrame,
    coad_msi_path: str | Path,
    ucec_msi_path: str | Path,
    stad_msi_path: str | Path,
    tmb_dir: str | Path,
    out_dir: str | Path,
    feature_name: str,
    max_folds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_cols = {"patient_id", "cancer_type", "OS.time", "OS", "os", "os_time", "stage"}
    feature_cols = [column for column in feature_df.columns if column not in metadata_cols]
    features = feature_df[["patient_id", "cancer_type", *feature_cols]].copy()
    features[feature_cols] = features[feature_cols].apply(pd.to_numeric, errors="coerce")

    labels = load_msi_labels(coad_msi_path, ucec_msi_path, stad_msi_path)
    tmb = load_tmb(tmb_dir)
    data = (
        features.merge(labels, on="patient_id", how="inner", validate="one_to_one")
        .merge(tmb, on=["patient_id", "cancer_type"], how="inner", validate="one_to_one")
    )
    data["msi_label"] = (data["msi_status"] == "MSI-H").astype(int)

    metric_rows = []
    prediction_frames = []
    for cohort, cancer_types in COHORTS.items():
        cohort_data = (
            data[data["cancer_type"].isin(cancer_types)]
            .sort_values("patient_id")
            .reset_index(drop=True)
        )
        class_counts = cohort_data["msi_label"].value_counts()
        if len(class_counts) < 2:
            raise ValueError(f"{cohort} does not contain both MSI-H and MSS patients.")
        n_splits = min(max_folds, int(class_counts.min()))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(splitter.split(cohort_data, cohort_data["msi_label"]))

        model_columns = {
            "TMB only": ["TMB_mutations_per_mb"],
            f"{feature_name} features": feature_cols,
        }
        for model_name, columns in model_columns.items():
            metrics, probabilities = evaluate_model(cohort_data, columns, splits)
            metric_rows.append(
                {
                    "cohort": cohort,
                    "model": model_name,
                    "n_folds": n_splits,
                    **metrics,
                }
            )
            prediction_frames.append(
                cohort_data[
                    ["patient_id", "cancer_type", "msi_status", "msi_label"]
                ].assign(
                    cohort=cohort,
                    model=model_name,
                    predicted_probability=probabilities,
                )
            )

        model_name = f"{feature_name} features residualized against TMB"
        metrics, probabilities = evaluate_residualized_model(
            cohort_data,
            feature_cols,
            splits,
        )
        metric_rows.append(
            {
                "cohort": cohort,
                "model": model_name,
                "n_folds": n_splits,
                **metrics,
            }
        )
        prediction_frames.append(
            cohort_data[
                ["patient_id", "cancer_type", "msi_status", "msi_label"]
            ].assign(
                cohort=cohort,
                model=model_name,
                predicted_probability=probabilities,
            )
        )

    results = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    results.to_csv(out_dir / "msi_classifier_metrics.tsv", sep="\t", index=False)
    predictions.to_csv(out_dir / "msi_classifier_predictions.tsv", sep="\t", index=False)

    return results, predictions

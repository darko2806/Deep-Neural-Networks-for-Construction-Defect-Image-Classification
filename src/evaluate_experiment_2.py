import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.feature_data import load_feature_splits


FEATURE_EXTRACTOR = "inception_resnet_v2"

VALIDATION_RESULT_PATH = Path(
    "data/results/xgboost/"
    "inception_resnet_v2_validation.json"
)

OUTPUT_DIRECTORY = Path(
    "data/results/experiment_2_multiclass"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Finalna evaluacija pobedničkog modela "
            "za Eksperiment 2"
        )
    )

    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )

    return parser.parse_args()


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def save_confusion_matrix_plot(
    counts,
    normalized,
    class_labels,
    output_path,
):
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
    )

    count_display = ConfusionMatrixDisplay(
        confusion_matrix=counts,
        display_labels=class_labels,
    )
    count_display.plot(
        ax=axes[0],
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )
    axes[0].set_title(
        "Test confusion matrix – broj slika"
    )

    normalized_display = ConfusionMatrixDisplay(
        confusion_matrix=normalized,
        display_labels=class_labels,
    )
    normalized_display.plot(
        ax=axes[1],
        cmap="Blues",
        colorbar=False,
        values_format=".2f",
    )
    axes[1].set_title(
        "Test confusion matrix – normalizovano"
    )

    for axis in axes:
        plt.setp(
            axis.get_xticklabels(),
            rotation=45,
            ha="right",
        )

    figure.suptitle(
        "Eksperiment 2 – "
        "Inception-ResNet-v2 + XGBoost",
        fontsize=14,
    )
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def main():
    args = parse_arguments()

    with VALIDATION_RESULT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        validation_result = json.load(file)

    best_parameters = validation_result[
        "best_parameters"
    ]

    feature_path = Path(
        f"data/features/{FEATURE_EXTRACTOR}.npz"
    )

    feature_splits = load_feature_splits(
        feature_path
    )

    X_train = feature_splits["train"]["X"]
    y_train = feature_splits["train"]["y"]

    X_validation = feature_splits[
        "validation"
    ]["X"]
    y_validation = feature_splits[
        "validation"
    ]["y"]

    X_test = feature_splits["test"]["X"]
    y_test_text = feature_splits["test"]["y"]
    test_image_ids = feature_splits[
        "test"
    ]["image_ids"]

    X_development = np.concatenate(
        [
            X_train,
            X_validation,
        ],
        axis=0,
    )
    y_development_text = np.concatenate(
        [
            y_train,
            y_validation,
        ],
        axis=0,
    )

    label_encoder = LabelEncoder()

    y_development = label_encoder.fit_transform(
        y_development_text
    )
    y_test = label_encoder.transform(
        y_test_text
    )

    class_labels = (
        label_encoder.classes_.astype(str)
    )

    if best_parameters["balanced_weights"]:
        development_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_development,
        ).astype(np.float32)
    else:
        development_weights = None

    classifier = XGBClassifier(
        objective="multi:softprob",
        num_class=len(class_labels),
        n_estimators=best_parameters[
            "n_estimators"
        ],
        learning_rate=best_parameters[
            "learning_rate"
        ],
        max_depth=best_parameters[
            "max_depth"
        ],
        min_child_weight=1,
        subsample=best_parameters["subsample"],
        colsample_bytree=best_parameters[
            "colsample_bytree"
        ],
        reg_alpha=best_parameters["reg_alpha"],
        reg_lambda=best_parameters["reg_lambda"],
        tree_method="hist",
        device=args.device,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )

    print("Eksperiment 2 – finalna evaluacija")
    print(
        "Model: Inception-ResNet-v2 + XGBoost"
    )
    print("Uređaj:", args.device)
    print(
        "Train + validation oblik:",
        X_development.shape,
    )
    print("Test oblik:", X_test.shape)
    print(
        "Hiperparametri:",
        best_parameters,
    )

    training_start = time.perf_counter()

    classifier.fit(
        X_development,
        y_development,
        sample_weight=development_weights,
    )

    training_time = (
        time.perf_counter()
        - training_start
    )

    prediction_start = time.perf_counter()

    test_probabilities = classifier.predict_proba(
        X_test
    )

    prediction_time = (
        time.perf_counter()
        - prediction_start
    )

    test_predictions_encoded = np.argmax(
        test_probabilities,
        axis=1,
    )

    test_predictions_text = (
        label_encoder.inverse_transform(
            test_predictions_encoded
        )
    )

    metrics = calculate_metrics(
        y_test_text,
        test_predictions_text,
    )

    report = classification_report(
        y_test_text,
        test_predictions_text,
        labels=class_labels,
        output_dict=True,
        zero_division=0,
    )

    confusion_counts = confusion_matrix(
        y_test_text,
        test_predictions_text,
        labels=class_labels,
    )

    row_totals = confusion_counts.sum(
        axis=1,
        keepdims=True,
    )

    confusion_normalized = np.divide(
        confusion_counts,
        row_totals,
        out=np.zeros_like(
            confusion_counts,
            dtype=np.float64,
        ),
        where=row_totals != 0,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        OUTPUT_DIRECTORY
        / "final_inception_resnet_v2_xgboost.json"
    )
    result_path = (
        OUTPUT_DIRECTORY
        / "final_test_results.json"
    )
    predictions_path = (
        OUTPUT_DIRECTORY
        / "test_predictions.csv"
    )
    report_path = (
        OUTPUT_DIRECTORY
        / "test_classification_report.csv"
    )
    confusion_counts_path = (
        OUTPUT_DIRECTORY
        / "test_confusion_matrix_counts.csv"
    )
    confusion_normalized_path = (
        OUTPUT_DIRECTORY
        / "test_confusion_matrix_normalized.csv"
    )
    confusion_plot_path = (
        OUTPUT_DIRECTORY
        / "test_confusion_matrix.png"
    )

    classifier.save_model(model_path)

    prediction_table = pd.DataFrame(
        {
            "image_id": test_image_ids.astype(str),
            "true_class": y_test_text.astype(str),
            "predicted_class": (
                test_predictions_text.astype(str)
            ),
            "confidence": test_probabilities.max(
                axis=1
            ),
            "correct": (
                y_test_text
                == test_predictions_text
            ),
        }
    )

    for class_index, class_name in enumerate(
        class_labels
    ):
        prediction_table[
            f"probability_{class_name}"
        ] = test_probabilities[:, class_index]

    prediction_table.to_csv(
        predictions_path,
        index=False,
    )

    report_table = pd.DataFrame(
        report
    ).transpose()

    report_table.to_csv(
        report_path,
        index=True,
    )

    confusion_counts_table = pd.DataFrame(
        confusion_counts,
        index=class_labels,
        columns=class_labels,
    )
    confusion_counts_table.index.name = "true_class"
    confusion_counts_table.columns.name = (
        "predicted_class"
    )
    confusion_counts_table.to_csv(
        confusion_counts_path
    )

    confusion_normalized_table = pd.DataFrame(
        confusion_normalized,
        index=class_labels,
        columns=class_labels,
    )
    confusion_normalized_table.index.name = (
        "true_class"
    )
    confusion_normalized_table.columns.name = (
        "predicted_class"
    )
    confusion_normalized_table.to_csv(
        confusion_normalized_path
    )

    save_confusion_matrix_plot(
        confusion_counts,
        confusion_normalized,
        class_labels,
        confusion_plot_path,
    )

    final_result = {
        "experiment": (
            "experiment_2_multiclass_"
            "defect_classification"
        ),
        "feature_extractor": FEATURE_EXTRACTOR,
        "classifier": "xgboost",
        "selection_metric": "validation_macro_f1",
        "validation_macro_f1": validation_result[
            "best_validation_macro_f1"
        ],
        "training_data": "train_plus_validation",
        "test_evaluations": 1,
        "device": args.device,
        "class_labels": class_labels.tolist(),
        "sample_counts": {
            "train": int(len(X_train)),
            "validation": int(
                len(X_validation)
            ),
            "train_plus_validation": int(
                len(X_development)
            ),
            "test": int(len(X_test)),
        },
        "parameters": best_parameters,
        "training_time_seconds": float(
            training_time
        ),
        "prediction_time_seconds": float(
            prediction_time
        ),
        "test_metrics": metrics,
        "classification_report": report,
    }

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_result,
            file,
            indent=2,
        )

    print("\nFinalni test rezultat:")
    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )
    print(
        f"Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Macro F1: "
        f"{metrics['macro_f1']:.4f}"
    )
    print(
        f"Weighted F1: "
        f"{metrics['weighted_f1']:.4f}"
    )

    print("\nRezultat po klasama:")
    print(
        report_table[
            [
                "precision",
                "recall",
                "f1-score",
                "support",
            ]
        ].to_string()
    )

    print("\nSačuvani fajlovi:")
    print(result_path)
    print(predictions_path)
    print(report_path)
    print(confusion_counts_path)
    print(confusion_normalized_path)
    print(confusion_plot_path)
    print(model_path)


if __name__ == "__main__":
    main()
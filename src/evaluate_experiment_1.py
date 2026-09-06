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
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.feature_data import (
    load_feature_splits,
)


FEATURE_EXTRACTOR = "resnet50"

FEATURE_PATH = Path(
    "data/features/experiment_1_binary/"
    "resnet50.npz"
)

VALIDATION_RESULT_PATH = Path(
    "data/results/experiment_1_binary/"
    "xgboost/resnet50_validation.json"
)

PATCH_METADATA_PATH = Path(
    "data/patch_metadata.csv"
)

OUTPUT_DIRECTORY = Path(
    "data/results/experiment_1_binary"
)

REPORT_CLASS_ORDER = [
    "no_defect",
    "defect",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Finalna evaluacija pobedničkog "
            "modela za Eksperiment 1."
        )
    )

    parser.add_argument(
        "--device",
        choices=[
            "cuda",
            "cpu",
        ],
        default="cuda",
    )

    return parser.parse_args()


def calculate_metrics(
    y_true,
    y_predicted,
    defect_probabilities,
):
    y_true_binary = (
        np.asarray(y_true) == "defect"
    ).astype(np.int64)

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_predicted,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_predicted,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_predicted,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_predicted,
                average="weighted",
                zero_division=0,
            )
        ),
        "defect_precision": float(
            precision_score(
                y_true,
                y_predicted,
                pos_label="defect",
                zero_division=0,
            )
        ),
        "defect_recall": float(
            recall_score(
                y_true,
                y_predicted,
                pos_label="defect",
                zero_division=0,
            )
        ),
        "defect_f1": float(
            f1_score(
                y_true,
                y_predicted,
                pos_label="defect",
                average="binary",
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true_binary,
                defect_probabilities,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_true_binary,
                defect_probabilities,
            )
        ),
    }


def save_confusion_matrix_plot(
    counts,
    normalized,
    output_path,
):
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13, 5.5),
    )

    counts_display = (
        ConfusionMatrixDisplay(
            confusion_matrix=counts,
            display_labels=(
                REPORT_CLASS_ORDER
            ),
        )
    )

    counts_display.plot(
        ax=axes[0],
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )

    axes[0].set_title(
        "Test confusion matrix – broj patch-eva"
    )

    normalized_display = (
        ConfusionMatrixDisplay(
            confusion_matrix=normalized,
            display_labels=(
                REPORT_CLASS_ORDER
            ),
        )
    )

    normalized_display.plot(
        ax=axes[1],
        cmap="Blues",
        colorbar=False,
        values_format=".3f",
    )

    axes[1].set_title(
        "Test confusion matrix – normalizovano"
    )

    for axis in axes:
        axis.set_xlabel(
            "Predikovana klasa"
        )
        axis.set_ylabel(
            "Stvarna klasa"
        )

        plt.setp(
            axis.get_xticklabels(),
            rotation=25,
            ha="right",
        )

    figure.suptitle(
        (
            "Eksperiment 1 – "
            "ResNet50 + XGBoost"
        ),
        fontsize=14,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def load_test_metadata(test_patch_ids):
    metadata = pd.read_csv(
        PATCH_METADATA_PATH
    )

    if metadata[
        "patch_id"
    ].duplicated().any():
        raise ValueError(
            "patch_id vrednosti nisu jedinstvene."
        )

    metadata_by_patch = (
        metadata
        .set_index("patch_id")
    )

    missing_patch_ids = (
        set(test_patch_ids)
        - set(metadata_by_patch.index)
    )

    if missing_patch_ids:
        raise ValueError(
            "Nedostaju metadata redovi za "
            f"{len(missing_patch_ids)} patch-eva."
        )

    test_metadata = (
        metadata_by_patch
        .loc[test_patch_ids]
        .reset_index()
    )

    if not (
        test_metadata["split"] == "test"
    ).all():
        raise ValueError(
            "Pronađen je patch koji ne pripada "
            "test splitu."
        )

    return test_metadata


def main():
    arguments = parse_arguments()

    required_paths = [
        FEATURE_PATH,
        VALIDATION_RESULT_PATH,
        PATCH_METADATA_PATH,
    ]

    for required_path in required_paths:
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Nedostaje fajl: {required_path}"
            )

    with VALIDATION_RESULT_PATH.open(
        "r",
        encoding="utf-8",
    ) as validation_file:
        validation_result = json.load(
            validation_file
        )

    if (
        validation_result["classifier"]
        != "xgboost"
    ):
        raise ValueError(
            "Pobednički klasifikator mora "
            "biti XGBoost."
        )

    if (
        validation_result["feature_extractor"]
        != FEATURE_EXTRACTOR
    ):
        raise ValueError(
            "Validation rezultat ne odgovara "
            "ResNet50 extractor-u."
        )

    best_parameters = validation_result[
        "best_parameters"
    ]

    feature_splits = load_feature_splits(
        FEATURE_PATH
    )

    X_train = feature_splits["train"]["X"]
    y_train_text = (
        feature_splits["train"]["y"]
    )

    X_validation = (
        feature_splits["validation"]["X"]
    )
    y_validation_text = (
        feature_splits["validation"]["y"]
    )

    X_test = feature_splits["test"]["X"]
    y_test_text = (
        feature_splits["test"]["y"]
    )

    test_patch_ids = (
        feature_splits["test"]["image_ids"]
        .astype(str)
    )

    X_development = np.concatenate(
        [
            X_train,
            X_validation,
        ],
        axis=0,
    )

    y_development_text = np.concatenate(
        [
            y_train_text,
            y_validation_text,
        ],
        axis=0,
    )

    label_encoder = LabelEncoder()

    y_development = (
        label_encoder.fit_transform(
            y_development_text
        )
    )

    if set(
        label_encoder.classes_
    ) != {
        "defect",
        "no_defect",
    }:
        raise ValueError(
            "Pronađene su neočekivane klase."
        )

    y_test_encoded = (
        label_encoder.transform(
            y_test_text
        )
    )

    classifier = XGBClassifier(
        objective="binary:logistic",
        n_estimators=best_parameters[
            "n_estimators"
        ],
        learning_rate=best_parameters[
            "learning_rate"
        ],
        max_depth=best_parameters[
            "max_depth"
        ],
        min_child_weight=best_parameters[
            "min_child_weight"
        ],
        subsample=best_parameters[
            "subsample"
        ],
        colsample_bytree=best_parameters[
            "colsample_bytree"
        ],
        reg_alpha=best_parameters[
            "reg_alpha"
        ],
        reg_lambda=best_parameters[
            "reg_lambda"
        ],
        tree_method="hist",
        device=arguments.device,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    print(
        "Eksperiment 1 – finalna evaluacija"
    )
    print(
        "Model: ResNet50 + XGBoost"
    )
    print("Uređaj:", arguments.device)
    print(
        "Train + validation oblik:",
        X_development.shape,
    )
    print(
        "Test oblik:",
        X_test.shape,
    )
    print(
        "Hiperparametri:",
        best_parameters,
    )

    training_start = time.perf_counter()

    classifier.fit(
        X_development,
        y_development,
    )

    training_time = (
        time.perf_counter()
        - training_start
    )

    prediction_start = time.perf_counter()

    test_probabilities = (
        classifier.predict_proba(
            X_test
        )
    )

    test_predictions_encoded = (
        classifier.predict(
            X_test
        ).astype(np.int64)
    )

    prediction_time = (
        time.perf_counter()
        - prediction_start
    )

    test_predictions_text = (
        label_encoder.inverse_transform(
            test_predictions_encoded
        )
    )

    defect_class_index = int(
        np.flatnonzero(
            label_encoder.classes_
            == "defect"
        )[0]
    )

    no_defect_class_index = int(
        np.flatnonzero(
            label_encoder.classes_
            == "no_defect"
        )[0]
    )

    defect_probabilities = (
        test_probabilities[
            :,
            defect_class_index,
        ]
    )

    no_defect_probabilities = (
        test_probabilities[
            :,
            no_defect_class_index,
        ]
    )

    metrics = calculate_metrics(
        y_test_text,
        test_predictions_text,
        defect_probabilities,
    )

    report = classification_report(
        y_test_text,
        test_predictions_text,
        labels=REPORT_CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )

    confusion_counts = confusion_matrix(
        y_test_text,
        test_predictions_text,
        labels=REPORT_CLASS_ORDER,
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

    test_metadata = load_test_metadata(
        test_patch_ids
    )

    if not np.array_equal(
        test_metadata[
            "target_class"
        ].to_numpy(dtype=str),
        y_test_text.astype(str),
    ):
        raise ValueError(
            "Redosled test metadata redova "
            "ne odgovara feature fajlu."
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = OUTPUT_DIRECTORY / (
        "final_resnet50_xgboost.json"
    )

    result_path = OUTPUT_DIRECTORY / (
        "final_test_results.json"
    )

    predictions_path = OUTPUT_DIRECTORY / (
        "test_predictions.csv"
    )

    errors_path = OUTPUT_DIRECTORY / (
        "test_misclassifications.csv"
    )

    report_path = OUTPUT_DIRECTORY / (
        "test_classification_report.csv"
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

    classifier.save_model(
        model_path
    )

    prediction_table = pd.DataFrame(
        {
            "patch_id": test_patch_ids,
            "pair_id": (
                test_metadata["pair_id"]
                .to_numpy(dtype=str)
            ),
            "image_id": (
                test_metadata["image_id"]
                .to_numpy(dtype=str)
            ),
            "image_path": (
                test_metadata["image_path"]
                .to_numpy(dtype=str)
            ),
            "source_defect_class": (
                test_metadata[
                    "source_defect_class"
                ].to_numpy(dtype=str)
            ),
            "xmin": (
                test_metadata["xmin"]
                .to_numpy(dtype=int)
            ),
            "ymin": (
                test_metadata["ymin"]
                .to_numpy(dtype=int)
            ),
            "xmax": (
                test_metadata["xmax"]
                .to_numpy(dtype=int)
            ),
            "ymax": (
                test_metadata["ymax"]
                .to_numpy(dtype=int)
            ),
            "true_class": (
                y_test_text.astype(str)
            ),
            "predicted_class": (
                test_predictions_text.astype(str)
            ),
            "probability_defect": (
                defect_probabilities
            ),
            "probability_no_defect": (
                no_defect_probabilities
            ),
            "confidence": (
                test_probabilities.max(
                    axis=1
                )
            ),
            "correct": (
                y_test_text
                == test_predictions_text
            ),
        }
    )

    prediction_table.to_csv(
        predictions_path,
        index=False,
    )

    prediction_table.loc[
        ~prediction_table["correct"]
    ].to_csv(
        errors_path,
        index=False,
    )

    report_table = (
        pd.DataFrame(report)
        .transpose()
    )

    report_table.to_csv(
        report_path,
        index=True,
    )

    confusion_counts_table = pd.DataFrame(
        confusion_counts,
        index=REPORT_CLASS_ORDER,
        columns=REPORT_CLASS_ORDER,
    )

    confusion_counts_table.index.name = (
        "true_class"
    )

    confusion_counts_table.columns.name = (
        "predicted_class"
    )

    confusion_counts_table.to_csv(
        confusion_counts_path
    )

    confusion_normalized_table = (
        pd.DataFrame(
            confusion_normalized,
            index=REPORT_CLASS_ORDER,
            columns=REPORT_CLASS_ORDER,
        )
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
        counts=confusion_counts,
        normalized=confusion_normalized,
        output_path=confusion_plot_path,
    )

    final_result = {
        "experiment": (
            "experiment_1_patch_binary"
        ),
        "task": (
            "defect_vs_no_annotated_defect_patch"
        ),
        "feature_extractor": (
            FEATURE_EXTRACTOR
        ),
        "classifier": "xgboost",
        "selection_metric": (
            "validation_macro_f1"
        ),
        "validation_macro_f1": (
            validation_result[
                "best_validation_macro_f1"
            ]
        ),
        "training_data": (
            "train_plus_validation"
        ),
        "test_evaluations": 1,
        "device": arguments.device,
        "class_labels": (
            REPORT_CLASS_ORDER
        ),
        "sample_counts": {
            "train": int(
                len(X_train)
            ),
            "validation": int(
                len(X_validation)
            ),
            "train_plus_validation": int(
                len(X_development)
            ),
            "test": int(
                len(X_test)
            ),
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
        "limitations": [
            (
                "no_defect označava patch koji "
                "ne preseca anotirani region"
            ),
            (
                "odsustvo anotacije ne garantuje "
                "potpuno zdravu površinu"
            ),
            (
                "rezultat predstavlja patch-level, "
                "a ne whole-image klasifikaciju"
            ),
        ],
    }

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as result_file:
        json.dump(
            final_result,
            result_file,
            indent=2,
        )

    print("\nFinalni test rezultat:")
    print(
        "Accuracy:",
        f"{metrics['accuracy']:.4f}",
    )
    print(
        "Balanced accuracy:",
        f"{metrics['balanced_accuracy']:.4f}",
    )
    print(
        "Macro F1:",
        f"{metrics['macro_f1']:.4f}",
    )
    print(
        "Defect precision:",
        f"{metrics['defect_precision']:.4f}",
    )
    print(
        "Defect recall:",
        f"{metrics['defect_recall']:.4f}",
    )
    print(
        "Defect F1:",
        f"{metrics['defect_f1']:.4f}",
    )
    print(
        "ROC AUC:",
        f"{metrics['roc_auc']:.4f}",
    )
    print(
        "Average precision:",
        f"{metrics['average_precision']:.4f}",
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

    print("\nConfusion matrix:")
    print(confusion_counts_table)

    print("\nSačuvani fajlovi:")
    print(result_path)
    print(predictions_path)
    print(errors_path)
    print(report_path)
    print(confusion_counts_path)
    print(confusion_normalized_path)
    print(confusion_plot_path)
    print(model_path)


if __name__ == "__main__":
    main()
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


FEATURE_PATH = Path(
    "data/features/experiment_2b_augmentation/"
    "inception_resnet_v2_merged.npz"
)
ORIGINAL_FEATURE_PATH = Path(
    "data/features/experiment_2_multiclass/"
    "inception_resnet_v2.npz"
)
VALIDATION_RESULT_PATH = Path(
    "data/results/experiment_2b_augmentation/"
    "xgboost_augmented_validation.json"
)
BASELINE_TEST_RESULT_PATH = Path(
    "data/results/experiment_2_multiclass/"
    "final_test_results.json"
)

OUTPUT_DIRECTORY = Path(
    "data/results/experiment_2b_augmentation"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Finalna test evaluacija Eksperimenta 2B"
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

    ConfusionMatrixDisplay(
        confusion_matrix=counts,
        display_labels=class_labels,
    ).plot(
        ax=axes[0],
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )
    axes[0].set_title(
        "Test confusion matrix – broj slika"
    )

    ConfusionMatrixDisplay(
        confusion_matrix=normalized,
        display_labels=class_labels,
    ).plot(
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
        "Eksperiment 2B – augmentacija + "
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


def save_comparison_markdown(
    comparison,
    class_comparison,
    output_path,
):
    lines = [
        "# Eksperiment 2B – poređenje test rezultata",
        "",
        (
            "Eksperiment 2B koristi train-only "
            "augmentaciju klasa bulge, corrosion i leakage. "
            "Model je izabran isključivo prema validationlob "
            "macro F1 rezultatu."
        ).replace("validationlob", "validation"),
        "",
        (
            "Isti test skup je korišćen i u osnovnom "
            "Eksperimentu 2, pa se rezultat Eksperimenta 2B "
            "tumači kao naknadna analiza."
        ),
        "",
        (
            "| Eksperiment | Validation macro F1 | "
            "Test accuracy | Test balanced accuracy | "
            "Test macro F1 | Test weighted F1 | "
            "Validation–test razlika |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.experiment} "
            f"| {row.validation_macro_f1:.4f} "
            f"| {row.test_accuracy:.4f} "
            f"| {row.test_balanced_accuracy:.4f} "
            f"| {row.test_macro_f1:.4f} "
            f"| {row.test_weighted_f1:.4f} "
            f"| {row.validation_test_gap:.4f} |"
        )

    lines.extend(
        [
            "",
            "## F1 rezultat po klasama",
            "",
            (
                "| Klasa | Eksperiment 2 | "
                "Eksperiment 2B | Promena |"
            ),
            "|---|---:|---:|---:|",
        ]
    )

    for row in class_comparison.itertuples(
        index=False
    ):
        lines.append(
            f"| {row.class_name} "
            f"| {row.baseline_f1:.4f} "
            f"| {row.augmented_f1:.4f} "
            f"| {row.change:+.4f} |"
        )

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_arguments()

    required_paths = [
        FEATURE_PATH,
        ORIGINAL_FEATURE_PATH,
        VALIDATION_RESULT_PATH,
        BASELINE_TEST_RESULT_PATH,
    ]

    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Nedostaje fajl: {path}"
            )

    with VALIDATION_RESULT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        validation_result = json.load(file)

    with BASELINE_TEST_RESULT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        baseline_result = json.load(file)

    best_parameters = validation_result[
        "best_parameters"
    ]

    augmented_splits = load_feature_splits(
        FEATURE_PATH
    )
    original_splits = load_feature_splits(
        ORIGINAL_FEATURE_PATH
    )

    # Dokaz da validation i test nisu augmentirani.
    for split_name in ("validation", "test"):
        augmented_split = augmented_splits[
            split_name
        ]
        original_split = original_splits[
            split_name
        ]

        if not np.array_equal(
            augmented_split["image_ids"],
            original_split["image_ids"],
        ):
            raise ValueError(
                f"{split_name} image ID-jevi su promenjeni"
            )

        if not np.array_equal(
            augmented_split["y"],
            original_split["y"],
        ):
            raise ValueError(
                f"{split_name} klase su promenjene"
            )

        if not np.allclose(
            augmented_split["X"],
            original_split["X"],
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError(
                f"{split_name} feature-i nisu isti kao originalni"
            )

    X_train = augmented_splits["train"]["X"]
    y_train_text = augmented_splits[
        "train"
    ]["y"]

    X_validation = augmented_splits[
        "validation"
    ]["X"]
    y_validation_text = augmented_splits[
        "validation"
    ]["y"]

    X_test = augmented_splits["test"]["X"]
    y_test_text = augmented_splits["test"]["y"]
    test_image_ids = augmented_splits[
        "test"
    ]["image_ids"]

    X_development = np.concatenate(
        [X_train, X_validation],
        axis=0,
    )
    y_development_text = np.concatenate(
        [y_train_text, y_validation_text],
        axis=0,
    )

    label_encoder = LabelEncoder()
    y_development = label_encoder.fit_transform(
        y_development_text
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

    print(
        "Eksperiment 2B – finalna test evaluacija"
    )
    print(
        "Model: Inception-ResNet-v2 + XGBoost"
    )
    print("Uređaj:", args.device)
    print(
        "Augmentirani train:",
        X_train.shape,
    )
    print(
        "Originalni validation:",
        X_validation.shape,
    )
    print(
        "Train + validation:",
        X_development.shape,
    )
    print("Originalni test:", X_test.shape)
    print("Hiperparametri:", best_parameters)
    print(
        "Provera originalnog validation/test "
        "skupa: uspešna"
    )

    training_start = time.perf_counter()

    classifier.fit(
        X_development,
        y_development,
        sample_weight=development_weights,
    )

    training_time = (
        time.perf_counter() - training_start
    )

    prediction_start = time.perf_counter()

    test_probabilities = classifier.predict_proba(
        X_test
    )

    prediction_time = (
        time.perf_counter() - prediction_start
    )

    predictions_encoded = np.argmax(
        test_probabilities,
        axis=1,
    )
    predictions_text = (
        label_encoder.inverse_transform(
            predictions_encoded
        )
    )

    metrics = calculate_metrics(
        y_test_text,
        predictions_text,
    )

    report = classification_report(
        y_test_text,
        predictions_text,
        labels=class_labels,
        output_dict=True,
        zero_division=0,
    )

    counts = confusion_matrix(
        y_test_text,
        predictions_text,
        labels=class_labels,
    )

    row_totals = counts.sum(
        axis=1,
        keepdims=True,
    )
    normalized = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(
            counts,
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
        / "final_augmented_inception_resnet_v2_xgboost.json"
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
    counts_path = (
        OUTPUT_DIRECTORY
        / "test_confusion_matrix_counts.csv"
    )
    normalized_path = (
        OUTPUT_DIRECTORY
        / "test_confusion_matrix_normalized.csv"
    )
    plot_path = (
        OUTPUT_DIRECTORY
        / "test_confusion_matrix.png"
    )
    comparison_path = (
        OUTPUT_DIRECTORY
        / "test_experiment_comparison.csv"
    )
    class_comparison_path = (
        OUTPUT_DIRECTORY
        / "test_class_f1_comparison.csv"
    )
    markdown_path = (
        OUTPUT_DIRECTORY
        / "test_experiment_comparison.md"
    )

    classifier.save_model(model_path)

    prediction_table = pd.DataFrame(
        {
            "image_id": test_image_ids.astype(str),
            "true_class": y_test_text.astype(str),
            "predicted_class": (
                predictions_text.astype(str)
            ),
            "confidence": test_probabilities.max(
                axis=1
            ),
            "correct": (
                y_test_text == predictions_text
            ),
        }
    )

    for index, class_name in enumerate(
        class_labels
    ):
        prediction_table[
            f"probability_{class_name}"
        ] = test_probabilities[:, index]

    prediction_table.to_csv(
        predictions_path,
        index=False,
    )

    report_table = pd.DataFrame(
        report
    ).transpose()
    report_table.to_csv(report_path)

    counts_table = pd.DataFrame(
        counts,
        index=class_labels,
        columns=class_labels,
    )
    counts_table.index.name = "true_class"
    counts_table.columns.name = "predicted_class"
    counts_table.to_csv(counts_path)

    normalized_table = pd.DataFrame(
        normalized,
        index=class_labels,
        columns=class_labels,
    )
    normalized_table.index.name = "true_class"
    normalized_table.columns.name = (
        "predicted_class"
    )
    normalized_table.to_csv(normalized_path)

    save_confusion_matrix_plot(
        counts,
        normalized,
        class_labels,
        plot_path,
    )

    baseline_metrics = baseline_result[
        "test_metrics"
    ]
    baseline_validation_f1 = float(
        baseline_result["validation_macro_f1"]
    )
    augmented_validation_f1 = float(
        validation_result[
            "best_augmented_validation_macro_f1"
        ]   
    )      

    comparison = pd.DataFrame(
        [
            {
                "experiment": "Eksperiment 2",
                "validation_macro_f1": (
                    baseline_validation_f1
                ),
                "test_accuracy": baseline_metrics[
                    "accuracy"
                ],
                "test_balanced_accuracy": (
                    baseline_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "test_macro_f1": baseline_metrics[
                    "macro_f1"
                ],
                "test_weighted_f1": (
                    baseline_metrics[
                        "weighted_f1"
                    ]
                ),
                "validation_test_gap": (
                    baseline_validation_f1
                    - baseline_metrics["macro_f1"]
                ),
            },
            {
                "experiment": (
                    "Eksperiment 2B – augmentacija"
                ),
                "validation_macro_f1": (
                    augmented_validation_f1
                ),
                "test_accuracy": metrics["accuracy"],
                "test_balanced_accuracy": (
                    metrics["balanced_accuracy"]
                ),
                "test_macro_f1": metrics["macro_f1"],
                "test_weighted_f1": (
                    metrics["weighted_f1"]
                ),
                "validation_test_gap": (
                    augmented_validation_f1
                    - metrics["macro_f1"]
                ),
            },
        ]
    )

    class_rows = []

    for class_name in class_labels:
        baseline_f1 = float(
            baseline_result[
                "classification_report"
            ][class_name]["f1-score"]
        )
        augmented_f1 = float(
            report[class_name]["f1-score"]
        )

        class_rows.append(
            {
                "class_name": class_name,
                "baseline_f1": baseline_f1,
                "augmented_f1": augmented_f1,
                "change": (
                    augmented_f1 - baseline_f1
                ),
            }
        )

    class_comparison = pd.DataFrame(
        class_rows
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )
    class_comparison.to_csv(
        class_comparison_path,
        index=False,
    )

    save_comparison_markdown(
        comparison,
        class_comparison,
        markdown_path,
    )

    final_result = {
        "experiment": (
            "experiment_2b_train_only_augmentation"
        ),
        "relationship_to_baseline": (
            "post_hoc_follow_up_experiment"
        ),
        "test_set_previously_used": True,
        "selection_metric": "validation_macro_f1",
        "feature_extractor": (
            "inception_resnet_v2"
        ),
        "classifier": "xgboost",
        "training_data": (
            "augmented_train_plus_original_validation"
        ),
        "validation_and_test_augmented": False,
        "device": args.device,
        "class_labels": class_labels.tolist(),
        "parameters": best_parameters,
        "sample_counts": {
            "augmented_train": int(
                len(X_train)
            ),
            "original_validation": int(
                len(X_validation)
            ),
            "development_total": int(
                len(X_development)
            ),
            "original_test": int(len(X_test)),
        },
        "validation_macro_f1": (
            augmented_validation_f1
        ),
        "training_time_seconds": float(
            training_time
        ),
        "prediction_time_seconds": float(
            prediction_time
        ),
        "test_metrics": metrics,
        "classification_report": report,
        "baseline_comparison": {
            "baseline_test_macro_f1": float(
                baseline_metrics["macro_f1"]
            ),
            "augmented_test_macro_f1": (
                metrics["macro_f1"]
            ),
            "test_macro_f1_change": float(
                metrics["macro_f1"]
                - baseline_metrics["macro_f1"]
            ),
            "baseline_validation_test_gap": float(
                baseline_validation_f1
                - baseline_metrics["macro_f1"]
            ),
            "augmented_validation_test_gap": float(
                augmented_validation_f1
                - metrics["macro_f1"]
            ),
        },
    }

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nFinalni test rezultat:")
    print(
        f"Accuracy: {metrics['accuracy']:.4f}"
    )
    print(
        "Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Macro F1: {metrics['macro_f1']:.4f}"
    )
    print(
        f"Weighted F1: {metrics['weighted_f1']:.4f}"
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

    print("\nPoređenje sa Eksperimentom 2:")
    print(comparison.to_string(index=False))

    print("\nPromena F1 po klasama:")
    print(
        class_comparison.to_string(
            index=False
        )
    )

    print("\nSačuvani fajlovi:")
    print(result_path)
    print(predictions_path)
    print(report_path)
    print(counts_path)
    print(normalized_path)
    print(plot_path)
    print(comparison_path)
    print(class_comparison_path)
    print(markdown_path)
    print(model_path)


if __name__ == "__main__":
    main()
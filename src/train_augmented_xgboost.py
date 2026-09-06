import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
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

BASELINE_RESULT_PATH = Path(
    "data/results/xgboost/"
    "inception_resnet_v2_validation.json"
)

OUTPUT_DIRECTORY = Path(
    "data/results/experiment_2b_augmentation"
)
RESULT_PATH = (
    OUTPUT_DIRECTORY
    / "xgboost_augmented_validation.json"
)
COMPARISON_CSV_PATH = (
    OUTPUT_DIRECTORY
    / "validation_augmentation_comparison.csv"
)
COMPARISON_MD_PATH = (
    OUTPUT_DIRECTORY
    / "validation_augmentation_comparison.md"
)

MAX_DEPTH_VALUES = [4, 8]
LEARNING_RATE_VALUES = [0.03, 0.1]
BALANCED_WEIGHT_VALUES = [False, True]

MAX_ESTIMATORS = 1000
EARLY_STOPPING_ROUNDS = 50
RANDOM_STATE = 42


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "XGBoost evaluacija augmentiranog train skupa "
            "za Eksperiment 2B"
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


def parameter_combinations():
    for (
        max_depth,
        learning_rate,
        balanced_weights,
    ) in product(
        MAX_DEPTH_VALUES,
        LEARNING_RATE_VALUES,
        BALANCED_WEIGHT_VALUES,
    ):
        yield {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "balanced_weights": balanced_weights,
        }


def best_result_for_weight(
    validation_results,
    balanced_weights,
):
    candidates = [
        result
        for result in validation_results
        if result["balanced_weights"]
        == balanced_weights
    ]

    return max(
        candidates,
        key=lambda result: result["macro_f1"],
    )


def comparison_row(
    dataset_variant,
    result,
):
    return {
        "dataset_variant": dataset_variant,
        "balanced_weights": result[
            "balanced_weights"
        ],
        "max_depth": result["max_depth"],
        "learning_rate": result[
            "learning_rate"
        ],
        "n_estimators": (
            result["best_iteration"] + 1
        ),
        "accuracy": result["accuracy"],
        "balanced_accuracy": result[
            "balanced_accuracy"
        ],
        "macro_f1": result["macro_f1"],
        "weighted_f1": result["weighted_f1"],
    }


def create_markdown(comparison):
    lines = [
        "# Eksperiment 2B – uticaj augmentacije",
        "",
        (
            "Poređenje je izvršeno na istom originalnom "
            "validation skupu. Augmentacija je primenjena "
            "isključivo na train slike klasa bulge, "
            "corrosion i leakage."
        ),
        "",
        (
            "| Skup | Težine klasa | Max depth | "
            "Learning rate | Broj stabala | Accuracy | "
            "Balanced accuracy | Macro F1 | Weighted F1 |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.dataset_variant} "
            f"| {row.balanced_weights} "
            f"| {row.max_depth} "
            f"| {row.learning_rate} "
            f"| {row.n_estimators} "
            f"| {row.accuracy:.4f} "
            f"| {row.balanced_accuracy:.4f} "
            f"| {row.macro_f1:.4f} "
            f"| {row.weighted_f1:.4f} |"
        )

    winner = comparison.iloc[0]

    lines.extend(
        [
            "",
            "## Najbolji validation rezultat",
            "",
            (
                f"- Skup: **{winner['dataset_variant']}**"
            ),
            (
                "- Balansirane težine: "
                f"**{winner['balanced_weights']}**"
            ),
            (
                "- Validation macro F1: "
                f"**{winner['macro_f1']:.4f}**"
            ),
            "",
        ]
    )

    return "\n".join(lines)


def main():
    args = parse_arguments()

    if not FEATURE_PATH.is_file():
        raise FileNotFoundError(
            f"Nedostaje feature fajl: {FEATURE_PATH}"
        )

    if not BASELINE_RESULT_PATH.is_file():
        raise FileNotFoundError(
            "Nedostaje originalni validation rezultat: "
            f"{BASELINE_RESULT_PATH}"
        )

    feature_splits = load_feature_splits(
        FEATURE_PATH
    )

    X_train = feature_splits["train"]["X"]
    y_train_text = feature_splits["train"]["y"]

    X_validation = feature_splits[
        "validation"
    ]["X"]
    y_validation_text = feature_splits[
        "validation"
    ]["y"]

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(
        y_train_text
    )
    y_validation = label_encoder.transform(
        y_validation_text
    )

    balanced_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    ).astype(np.float32)

    configurations = list(
        parameter_combinations()
    )
    validation_results = []

    print(
        "Eksperiment 2B – XGBoost sa "
        "train-only augmentacijom"
    )
    print("Uređaj:", args.device)
    print("Train oblik:", X_train.shape)
    print(
        "Validation oblik:",
        X_validation.shape,
    )
    print(
        "Klase:",
        label_encoder.classes_.tolist(),
    )
    print(
        "Broj konfiguracija:",
        len(configurations),
    )

    unique_classes, class_counts = np.unique(
        y_train_text,
        return_counts=True,
    )

    print("\nTrain broj po klasama:")
    for class_name, count in zip(
        unique_classes,
        class_counts,
    ):
        print(f"{class_name}: {count}")

    for index, parameters in enumerate(
        configurations,
        start=1,
    ):
        print(
            f"\nKonfiguracija {index}/"
            f"{len(configurations)}"
        )
        print(
            f"max_depth={parameters['max_depth']}, "
            f"learning_rate="
            f"{parameters['learning_rate']}, "
            f"balanced_weights="
            f"{parameters['balanced_weights']}",
            flush=True,
        )

        classifier = XGBClassifier(
            objective="multi:softprob",
            num_class=len(
                label_encoder.classes_
            ),
            n_estimators=MAX_ESTIMATORS,
            learning_rate=parameters[
                "learning_rate"
            ],
            max_depth=parameters["max_depth"],
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            tree_method="hist",
            device=args.device,
            eval_metric="mlogloss",
            early_stopping_rounds=(
                EARLY_STOPPING_ROUNDS
            ),
            random_state=RANDOM_STATE,
            verbosity=0,
        )

        fit_arguments = {
            "X": X_train,
            "y": y_train,
            "eval_set": [
                (
                    X_validation,
                    y_validation,
                )
            ],
            "verbose": False,
        }

        if parameters["balanced_weights"]:
            fit_arguments["sample_weight"] = (
                balanced_weights
            )

        training_start = time.perf_counter()

        classifier.fit(**fit_arguments)

        training_time = (
            time.perf_counter()
            - training_start
        )

        prediction_start = time.perf_counter()

        predictions_encoded = classifier.predict(
            X_validation
        )

        prediction_time = (
            time.perf_counter()
            - prediction_start
        )

        predictions_text = (
            label_encoder.inverse_transform(
                predictions_encoded.astype(
                    np.int64
                )
            )
        )

        metrics = calculate_metrics(
            y_validation_text,
            predictions_text,
        )

        best_iteration = classifier.best_iteration

        result = {
            "max_depth": parameters["max_depth"],
            "learning_rate": parameters[
                "learning_rate"
            ],
            "balanced_weights": parameters[
                "balanced_weights"
            ],
            "maximum_estimators": MAX_ESTIMATORS,
            "best_iteration": (
                int(best_iteration)
                if best_iteration is not None
                else MAX_ESTIMATORS - 1
            ),
            "training_time_seconds": float(
                training_time
            ),
            "prediction_time_seconds": float(
                prediction_time
            ),
            **metrics,
        }

        validation_results.append(result)

        print(
            f"accuracy={metrics['accuracy']:.4f}, "
            f"balanced_accuracy="
            f"{metrics['balanced_accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}, "
            f"weighted_f1="
            f"{metrics['weighted_f1']:.4f}, "
            f"best_iteration="
            f"{result['best_iteration']}, "
            f"training_time={training_time:.1f}s"
        )

    augmented_best = max(
        validation_results,
        key=lambda result: result["macro_f1"],
    )

    with BASELINE_RESULT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        baseline_result = json.load(file)

    baseline_validation_results = baseline_result[
        "validation_results"
    ]

    comparison_rows = []

    for use_balanced_weights in (False, True):
        comparison_rows.append(
            comparison_row(
                dataset_variant="Originalni train",
                result=best_result_for_weight(
                    baseline_validation_results,
                    use_balanced_weights,
                ),
            )
        )

        comparison_rows.append(
            comparison_row(
                dataset_variant="Augmentirani train",
                result=best_result_for_weight(
                    validation_results,
                    use_balanced_weights,
                ),
            )
        )

    comparison = pd.DataFrame(
        comparison_rows
    ).sort_values(
        by="macro_f1",
        ascending=False,
    ).reset_index(drop=True)

    baseline_best = max(
        baseline_validation_results,
        key=lambda result: result["macro_f1"],
    )

    improvement = (
        augmented_best["macro_f1"]
        - baseline_best["macro_f1"]
    )

    experiment_result = {
        "experiment": (
            "experiment_2b_train_only_augmentation"
        ),
        "feature_extractor": (
            "inception_resnet_v2"
        ),
        "classifier": "xgboost",
        "device": args.device,
        "selection_metric": "validation_macro_f1",
        "sample_counts": {
            "train_with_augmentation": int(
                len(X_train)
            ),
            "validation_original": int(
                len(X_validation)
            ),
        },
        "class_labels": (
            label_encoder.classes_.tolist()
        ),
        "best_parameters": {
            "max_depth": augmented_best[
                "max_depth"
            ],
            "learning_rate": augmented_best[
                "learning_rate"
            ],
            "balanced_weights": augmented_best[
                "balanced_weights"
            ],
            "n_estimators": (
                augmented_best[
                    "best_iteration"
                ] + 1
            ),
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        "best_augmented_validation_macro_f1": (
            augmented_best["macro_f1"]
        ),
        "baseline_validation_macro_f1": (
            baseline_best["macro_f1"]
        ),
        "macro_f1_difference": float(
            improvement
        ),
        "augmentation_improved_validation": bool(
            improvement > 0
        ),
        "validation_results": (
            validation_results
        ),
    }

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            experiment_result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    comparison.to_csv(
        COMPARISON_CSV_PATH,
        index=False,
    )

    COMPARISON_MD_PATH.write_text(
        create_markdown(comparison),
        encoding="utf-8",
    )

    print("\nPoređenje najboljih konfiguracija:")
    print(comparison.to_string(index=False))

    print("\nNajbolji augmentirani rezultat:")
    print(
        "max_depth:",
        augmented_best["max_depth"],
    )
    print(
        "learning_rate:",
        augmented_best["learning_rate"],
    )
    print(
        "balanced_weights:",
        augmented_best["balanced_weights"],
    )
    print(
        "n_estimators:",
        augmented_best["best_iteration"] + 1,
    )
    print(
        "Macro F1:",
        f"{augmented_best['macro_f1']:.4f}",
    )

    print(
        "\nOriginalni najbolji Macro F1:",
        f"{baseline_best['macro_f1']:.4f}",
    )
    print(
        "Promena Macro F1:",
        f"{improvement:+.4f}",
    )

    if improvement > 0:
        print(
            "Zaključak: augmentirani model je "
            "pobedio na validation skupu."
        )
    else:
        print(
            "Zaključak: augmentacija nije poboljšala "
            "validation macro F1."
        )

    print("\nSačuvano:")
    print(RESULT_PATH)
    print(COMPARISON_CSV_PATH)
    print(COMPARISON_MD_PATH)


if __name__ == "__main__":
    main()
import argparse
import json
import time
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_data import (
    load_feature_splits,
)
from src.feature_extraction import (
    FEATURE_EXTRACTOR_MODELS,
)


C_VALUES = [
    0.01,
    0.1,
    1.0,
    10.0,
]

FEATURE_DIRECTORY = Path(
    "data/features/experiment_1_binary"
)

OUTPUT_DIRECTORY = Path(
    "data/results/experiment_1_binary/"
    "logistic_regression"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Logistička regresija za "
            "Eksperiment 1."
        )
    )

    parser.add_argument(
        "--model",
        choices=sorted(
            FEATURE_EXTRACTOR_MODELS
        ),
        required=True,
    )

    return parser.parse_args()


def calculate_metrics(
    y_true,
    y_predicted,
):
    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_predicted,
        ),
        "balanced_accuracy": (
            balanced_accuracy_score(
                y_true,
                y_predicted,
            )
        ),
        "macro_f1": f1_score(
            y_true,
            y_predicted,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_predicted,
            average="weighted",
            zero_division=0,
        ),
        "defect_f1": f1_score(
            y_true,
            y_predicted,
            pos_label="defect",
            average="binary",
            zero_division=0,
        ),
    }

    return {
        metric_name: float(metric_value)
        for metric_name, metric_value
        in metrics.items()
    }


def main():
    arguments = parse_arguments()

    feature_path = FEATURE_DIRECTORY / (
        f"{arguments.model}.npz"
    )

    output_path = OUTPUT_DIRECTORY / (
        f"{arguments.model}_validation.json"
    )

    feature_splits = load_feature_splits(
        feature_path
    )

    X_train = feature_splits["train"]["X"]
    y_train = feature_splits["train"]["y"]

    X_validation = (
        feature_splits["validation"]["X"]
    )
    y_validation = (
        feature_splits["validation"]["y"]
    )

    print(
        "Eksperiment 1 – patch-level "
        "binarna klasifikacija"
    )
    print(
        "Klasifikator: Logistic Regression"
    )
    print(
        "Feature extractor:",
        arguments.model,
    )
    print(
        "Train oblik:",
        X_train.shape,
    )
    print(
        "Validation oblik:",
        X_validation.shape,
    )

    validation_results = []

    for c_value in C_VALUES:
        print(
            f"\nTreniranje za C={c_value}...",
            flush=True,
        )

        pipeline = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value,
                        solver="lbfgs",
                        max_iter=2000,
                        class_weight=None,
                        random_state=42,
                    ),
                ),
            ]
        )

        start_time = time.perf_counter()

        pipeline.fit(
            X_train,
            y_train,
        )

        training_time = (
            time.perf_counter()
            - start_time
        )

        validation_predictions = (
            pipeline.predict(
                X_validation
            )
        )

        metrics = calculate_metrics(
            y_validation,
            validation_predictions,
        )

        number_of_iterations = int(
            pipeline
            .named_steps["classifier"]
            .n_iter_
            .max()
        )

        result = {
            "C": c_value,
            "training_time_seconds": float(
                training_time
            ),
            "n_iterations": (
                number_of_iterations
            ),
            **metrics,
        }

        validation_results.append(result)

        print(
            f"accuracy="
            f"{metrics['accuracy']:.4f}, "
            f"balanced_accuracy="
            f"{metrics['balanced_accuracy']:.4f}, "
            f"macro_f1="
            f"{metrics['macro_f1']:.4f}, "
            f"defect_f1="
            f"{metrics['defect_f1']:.4f}, "
            f"iterations="
            f"{number_of_iterations}"
        )

    best_result = max(
        validation_results,
        key=lambda result: (
            result["macro_f1"]
        ),
    )

    experiment_result = {
        "experiment": (
            "experiment_1_patch_binary"
        ),
        "classifier": (
            "logistic_regression"
        ),
        "feature_extractor": (
            arguments.model
        ),
        "selection_metric": "macro_f1",
        "best_C": best_result["C"],
        "best_validation_macro_f1": (
            best_result["macro_f1"]
        ),
        "validation_results": (
            validation_results
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            experiment_result,
            output_file,
            indent=2,
        )

    print("\nNajbolji validation rezultat:")
    print("C:", best_result["C"])
    print(
        "Accuracy:",
        f"{best_result['accuracy']:.4f}",
    )
    print(
        "Macro F1:",
        f"{best_result['macro_f1']:.4f}",
    )
    print(
        "Defect F1:",
        f"{best_result['defect_f1']:.4f}",
    )
    print("Sačuvano:", output_path)


if __name__ == "__main__":
    main()
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

from src.feature_data import load_feature_splits
from src.feature_extraction import FEATURE_EXTRACTOR_MODELS


C_VALUES = [0.01, 0.1, 1.0, 10.0]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Izbor Logistic Regression hiperparametra"
    )

    parser.add_argument(
        "--model",
        choices=sorted(FEATURE_EXTRACTOR_MODELS),
        required=True,
    )

    return parser.parse_args()


def calculate_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }

    return {
        metric_name: float(metric_value)
        for metric_name, metric_value in metrics.items()
    }


def main():
    args = parse_arguments()

    feature_path = Path(
        f"data/features/{args.model}.npz"
    )

    output_path = Path(
        f"data/results/logistic_regression/"
        f"{args.model}_validation.json"
    )

    feature_splits = load_feature_splits(feature_path)

    X_train = feature_splits["train"]["X"]
    y_train = feature_splits["train"]["y"]

    X_validation = feature_splits["validation"]["X"]
    y_validation = feature_splits["validation"]["y"]

    validation_results = []

    print(f"Feature extractor: {args.model}")
    print(f"Train oblik: {X_train.shape}")
    print(f"Validation oblik: {X_validation.shape}")

    for c_value in C_VALUES:
        print(
            f"\nTreniranje za C={c_value}...",
            flush=True,
        )

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
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

        pipeline.fit(X_train, y_train)

        training_time = time.perf_counter() - start_time

        validation_predictions = pipeline.predict(
            X_validation
        )

        metrics = calculate_metrics(
            y_validation,
            validation_predictions,
        )

        n_iterations = int(
            pipeline.named_steps["classifier"]
            .n_iter_
            .max()
        )

        result = {
            "C": c_value,
            "training_time_seconds": training_time,
            "n_iterations": n_iterations,
            **metrics,
        }

        validation_results.append(result)

        print(
            f"accuracy={metrics['accuracy']:.4f}, "
            f"balanced_accuracy="
            f"{metrics['balanced_accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}, "
            f"weighted_f1={metrics['weighted_f1']:.4f}, "
            f"iterations={n_iterations}"
        )

    best_result = max(
        validation_results,
        key=lambda result: result["macro_f1"],
    )

    experiment_result = {
        "classifier": "logistic_regression",
        "feature_extractor": args.model,
        "selection_metric": "macro_f1",
        "best_C": best_result["C"],
        "best_validation_macro_f1": best_result["macro_f1"],
        "validation_results": validation_results,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            experiment_result,
            file,
            indent=2,
        )

    print("\nNajbolji validation rezultat:")
    print(f"C: {best_result['C']}")
    print(
        f"Macro F1: {best_result['macro_f1']:.4f}"
    )
    print(f"Rezultati su sačuvani u: {output_path}")


if __name__ == "__main__":
    main()
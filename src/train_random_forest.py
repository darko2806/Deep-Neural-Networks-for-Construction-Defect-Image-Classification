import argparse
import json
import time
from itertools import product
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from src.feature_data import load_feature_splits
from src.feature_extraction import FEATURE_EXTRACTOR_MODELS


N_ESTIMATORS = 300

MAX_DEPTH_VALUES = [
    None,
    40,
]

MAX_FEATURES_VALUES = [
    "sqrt",
    0.25,
]

CLASS_WEIGHT_VALUES = [
    None,
    "balanced_subsample",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Izbor Random Forest hiperparametara "
            "na validation skupu"
        )
    )

    parser.add_argument(
        "--model",
        choices=sorted(FEATURE_EXTRACTOR_MODELS),
        required=True,
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help=(
            "Broj paralelnih CPU procesa. "
            "-1 koristi sva dostupna jezgra."
        ),
    )

    return parser.parse_args()


def calculate_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
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


def create_parameter_combinations():
    combinations = product(
        MAX_DEPTH_VALUES,
        MAX_FEATURES_VALUES,
        CLASS_WEIGHT_VALUES,
    )

    return [
        {
            "max_depth": max_depth,
            "max_features": max_features,
            "class_weight": class_weight,
        }
        for (
            max_depth,
            max_features,
            class_weight,
        ) in combinations
    ]


def main():
    args = parse_arguments()

    feature_path = Path(
        f"data/features/{args.model}.npz"
    )

    output_path = Path(
        "data/results/random_forest/"
        f"{args.model}_validation.json"
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

    parameter_combinations = (
        create_parameter_combinations()
    )

    validation_results = []

    print(f"Feature extractor: {args.model}")
    print(f"Train oblik: {X_train.shape}")
    print(
        f"Validation oblik: "
        f"{X_validation.shape}"
    )
    print(
        "Broj konfiguracija:",
        len(parameter_combinations),
    )

    for configuration_index, parameters in enumerate(
        parameter_combinations,
        start=1,
    ):
        print(
            f"\nKonfiguracija "
            f"{configuration_index}/"
            f"{len(parameter_combinations)}"
        )
        print(
            f"max_depth={parameters['max_depth']}, "
            f"max_features="
            f"{parameters['max_features']}, "
            f"class_weight="
            f"{parameters['class_weight']}",
            flush=True,
        )

        classifier = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=parameters["max_depth"],
            max_features=parameters["max_features"],
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight=parameters["class_weight"],
            criterion="gini",
            bootstrap=True,
            n_jobs=args.n_jobs,
            random_state=42,
        )

        training_start = time.perf_counter()

        classifier.fit(
            X_train,
            y_train,
        )

        training_time = (
            time.perf_counter()
            - training_start
        )

        prediction_start = time.perf_counter()

        validation_predictions = classifier.predict(
            X_validation
        )

        prediction_time = (
            time.perf_counter()
            - prediction_start
        )

        metrics = calculate_metrics(
            y_validation,
            validation_predictions,
        )

        result = {
            "n_estimators": N_ESTIMATORS,
            "max_depth": parameters["max_depth"],
            "max_features": parameters[
                "max_features"
            ],
            "class_weight": parameters[
                "class_weight"
            ],
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
            f"training_time="
            f"{training_time:.1f}s"
        )

    best_result = max(
        validation_results,
        key=lambda result: result["macro_f1"],
    )

    experiment_result = {
        "classifier": "random_forest",
        "feature_extractor": args.model,
        "selection_metric": "macro_f1",
        "best_parameters": {
            "n_estimators": best_result[
                "n_estimators"
            ],
            "max_depth": best_result[
                "max_depth"
            ],
            "max_features": best_result[
                "max_features"
            ],
            "class_weight": best_result[
                "class_weight"
            ],
        },
        "best_validation_macro_f1": (
            best_result["macro_f1"]
        ),
        "validation_results": validation_results,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            experiment_result,
            file,
            indent=2,
        )

    print("\nNajbolji validation rezultat:")
    print(
        "n_estimators:",
        best_result["n_estimators"],
    )
    print(
        "max_depth:",
        best_result["max_depth"],
    )
    print(
        "max_features:",
        best_result["max_features"],
    )
    print(
        "class_weight:",
        best_result["class_weight"],
    )
    print(
        "Macro F1:",
        f"{best_result['macro_f1']:.4f}",
    )
    print(
        "Rezultati su sačuvani u:",
        output_path,
    )


if __name__ == "__main__":
    main()
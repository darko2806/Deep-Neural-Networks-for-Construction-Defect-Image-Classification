import argparse
import json
import time
from itertools import product
from pathlib import Path

from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from src.feature_data import (
    load_feature_splits,
)
from src.feature_extraction import (
    FEATURE_EXTRACTOR_MODELS,
)


N_ESTIMATORS = 300

MAX_DEPTH_VALUES = [
    None,
    40,
]

MAX_FEATURES_VALUES = [
    "sqrt",
    0.25,
]

MIN_SAMPLES_LEAF_VALUES = [
    1,
    5,
]

FEATURE_DIRECTORY = Path(
    "data/features/experiment_1_binary"
)

OUTPUT_DIRECTORY = Path(
    "data/results/experiment_1_binary/"
    "random_forest"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Random Forest za patch-level "
            "binarnu klasifikaciju."
        )
    )

    parser.add_argument(
        "--model",
        choices=sorted(
            FEATURE_EXTRACTOR_MODELS
        ),
        required=True,
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
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


def create_parameter_combinations():
    combinations = product(
        MAX_DEPTH_VALUES,
        MAX_FEATURES_VALUES,
        MIN_SAMPLES_LEAF_VALUES,
    )

    return [
        {
            "max_depth": max_depth,
            "max_features": max_features,
            "min_samples_leaf": (
                min_samples_leaf
            ),
        }
        for (
            max_depth,
            max_features,
            min_samples_leaf,
        ) in combinations
    ]


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

    parameter_combinations = (
        create_parameter_combinations()
    )

    print(
        "Eksperiment 1 – patch-level "
        "binarna klasifikacija"
    )
    print("Klasifikator: Random Forest")
    print(
        "Feature extractor:",
        arguments.model,
    )
    print("Train oblik:", X_train.shape)
    print(
        "Validation oblik:",
        X_validation.shape,
    )
    print(
        "Broj konfiguracija:",
        len(parameter_combinations),
    )

    validation_results = []

    for configuration_index, parameters in (
        enumerate(
            parameter_combinations,
            start=1,
        )
    ):
        print(
            f"\nKonfiguracija "
            f"{configuration_index}/"
            f"{len(parameter_combinations)}"
        )

        print(
            f"max_depth="
            f"{parameters['max_depth']}, "
            f"max_features="
            f"{parameters['max_features']}, "
            f"min_samples_leaf="
            f"{parameters['min_samples_leaf']}",
            flush=True,
        )

        classifier = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=parameters[
                "max_depth"
            ],
            max_features=parameters[
                "max_features"
            ],
            min_samples_split=2,
            min_samples_leaf=parameters[
                "min_samples_leaf"
            ],
            class_weight=None,
            criterion="gini",
            bootstrap=True,
            n_jobs=arguments.n_jobs,
            random_state=42,
        )

        start_time = time.perf_counter()

        classifier.fit(
            X_train,
            y_train,
        )

        training_time = (
            time.perf_counter()
            - start_time
        )

        validation_predictions = (
            classifier.predict(
                X_validation
            )
        )

        metrics = calculate_metrics(
            y_validation,
            validation_predictions,
        )

        result = {
            "n_estimators": N_ESTIMATORS,
            "max_depth": parameters[
                "max_depth"
            ],
            "max_features": parameters[
                "max_features"
            ],
            "min_samples_leaf": parameters[
                "min_samples_leaf"
            ],
            "class_weight": None,
            "training_time_seconds": float(
                training_time
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
            f"training_time="
            f"{training_time:.1f}s"
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
        "classifier": "random_forest",
        "feature_extractor": (
            arguments.model
        ),
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
            "min_samples_leaf": best_result[
                "min_samples_leaf"
            ],
            "class_weight": None,
        },
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

    for parameter_name in (
        "n_estimators",
        "max_depth",
        "max_features",
        "min_samples_leaf",
    ):
        print(
            f"{parameter_name}:",
            best_result[parameter_name],
        )

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